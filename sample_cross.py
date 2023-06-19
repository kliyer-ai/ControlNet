from torch.utils.data import DataLoader
import einops
import numpy as np
import torch
from PIL import Image
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange
from kinetics import Kinetics700InterpolateBase


p = "./train_log/kin_hed_dropout1/lightning_logs/version_1/checkpoints/epoch=6-step=595385.ckpt"

model = create_model("./models/cldm_v15_cross.yaml").cuda()

model.load_state_dict(load_state_dict(p, location="cuda"))
ddim_sampler = DDIMSampler(model)
frozenClipImageEmbedder = model.style_encoder

ddim_steps = 40
strength = 1
eta = 0
scale = 7


def warp_frame(start_frame, flow, round=True):
    warped_frame = torch.ones_like(start_frame) * -1.0
    target_mask = torch.ones_like(start_frame)[:, 0] * -1.0
    target_mask = rearrange(target_mask, "b h w -> b 1 h w")
    flowlength = torch.sqrt(torch.sum(flow**2, dim=1))  # length of the flow vector
    flowlength = rearrange(flowlength, "n h w -> n (h w)")
    source_indices = torch.argsort(
        flowlength, dim=-1
    )  # sort by  warping pixels with small flow first
    xx, yy = torch.meshgrid(
        torch.arange(start_frame.size(2)),
        torch.arange(start_frame.size(3)),
        indexing="xy",
    )
    grid = (
        torch.cat([xx.unsqueeze(0), yy.unsqueeze(0)], dim=0)
        .unsqueeze(0)
        .repeat(start_frame.size(0), 1, 1, 1)
        .cuda()
    )

    if round:
        vgrid = grid + torch.round(flow).long()
    else:
        vgrid = grid + flow.long()

    maskw = torch.logical_and(
        vgrid[:, 0, :, :] >= 0, vgrid[:, 0, :, :] < start_frame.size(3)
    )
    maskh = torch.logical_and(
        vgrid[:, 1, :, :] >= 0, vgrid[:, 1, :, :] < start_frame.size(2)
    )
    mask = torch.logical_and(
        maskw, maskh
    )  # mask of pixels we are allowed to move to prevent out of domain mapping
    mask = rearrange(mask, "n h w -> n (h w)")

    for b in range(start_frame.size(0)):
        # filter indices
        filtered_source_indices = torch.masked_select(
            source_indices[b], mask[b, source_indices[b]]
        )  # only select source indices which don't map out of domain
        source_pixels = torch.index_select(
            rearrange(start_frame[b], "c h w -> c (h w)"), -1, filtered_source_indices
        )  # order pixels from source image
        target_indices = torch.index_select(
            rearrange(vgrid[b], "c h w -> c (h w)"), -1, filtered_source_indices
        )
        target_indices = (
            target_indices[1] * start_frame.size(3) + target_indices[0]
        )  # convert to flattened indices

        # create mask here for inpainting
        temp_mask = torch.ones((start_frame.size(2) * start_frame.size(3))).cuda() * 1.0
        temp_mask[
            target_indices
        ] = -1.0  # black pixels are kept, only white pixels should be masked regions
        target_mask[b] = rearrange(temp_mask, "(h w) -> 1 h w", h=start_frame.size(2))

        # set pixel at target_indices location
        temp = (
            torch.ones(
                (start_frame.size(1), start_frame.size(2) * start_frame.size(3))
            ).cuda()
            * -1.0
        )
        temp[:, target_indices] = source_pixels
        warped_frame[b] = rearrange(temp, "c (h w) -> c h w", h=start_frame.size(2))

    return warped_frame, target_mask  # warped_frame is (n c h w)


dataset = Kinetics700InterpolateBase(
    sequence_time=0.5,
    sequence_length=None,
    size=512,
    resize_size=None,
    random_crop=None,
    pixel_range=2,
    interpolation="bicubic",
    mode="val",
    data_path="/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020",
    dataset_size=1.0,
    filter_file="/export/home/koktay/flow_diffusion/scripts/timestamps_validation.json",
    flow_only=False,
    include_full_sequence=False,
    include_hed=True,
)

torch.manual_seed(42)
dl = DataLoader(dataset, shuffle=True, batch_size=4)
_iter = iter(dl)

for i in range(250):
    # styles, structures = get_sequence(dl)
    batch = next(_iter)

    # first element of batch (full sequence)
    styles = batch["start_frame"]

    styles = (styles + 1.0) * 127.5
    styles = styles.clip(0, 255).type(torch.uint8)
    # styles = einops.rearrange(styles, "b h w c -> b c h w").cuda()

    control = batch["hed_intermediate_frame"]
    control = einops.rearrange(control, "b h w c -> b c h w").cuda()

    B, C, H, W = control.shape

    style_embedding = frozenClipImageEmbedder(styles)

    c_style = style_embedding.last_hidden_state
    c_embed = style_embedding.pooler_output
    c_prompt = model.get_learned_conditioning(
        ["a professional, detailed, high-quality image"] * B
    )

    uc_style = torch.zeros_like(c_style)
    uc_embed = torch.zeros_like(c_embed)
    uc_prompt = model.get_learned_conditioning([""] * B)

    cond = {
        "c_concat": [control],
        "c_crossattn": [c_prompt],
        "c_style": [c_style],
        "c_embed": [c_embed],
    }
    un_cond = {
        "c_concat": [control],
        "c_crossattn": [uc_prompt],
        "c_style": [uc_style],
        "c_embed": [uc_embed],
    }
    shape = (4, H // 8, W // 8)

    # only need to potentially change this for guess mode
    model.control_scales = [strength] * 13

    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        B,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
    )

    x_samples = model.decode_first_stage(samples) * 127.5 + 127.5
    x_samples = einops.rearrange(x_samples, "b c h w -> b h w c")
    x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)

    for j in range(B):
        Image.fromarray(x_samples[j]).save(f"samples_cross/x/img_{i}-{j}.png")
        Image.fromarray(styles[j]).save(f"samples_cross/s/img_{i}-{j}.png")
