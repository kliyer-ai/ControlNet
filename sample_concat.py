from torch.utils.data import DataLoader
import einops
import numpy as np
import torch
from PIL import Image
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange
from kinetics import Kinetics700InterpolateBase


p = "./train_log/kin_hed_concat7/lightning_logs/version_0/checkpoints/epoch=0-step=54899.ckpt"

model = create_model("./models/cldm_v15_concat.yaml").cuda()

model.load_state_dict(load_state_dict(p, location="cuda"))
ddim_sampler = DDIMSampler(model)

ddim_steps = 40
strength = 1
eta = 0
scale = 3
batch_size = 4


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
    filter_file="./data_val.json",
    flow_only=False,
    include_full_sequence=False,
    include_hed=True,
)

torch.manual_seed(42)
dl = DataLoader(dataset, shuffle=False, batch_size=batch_size)
_iter = iter(dl)

for i in range(2000):
    # styles, structures = get_sequence(dl)
    batch = next(_iter)

    # first element of batch (full sequence)
    styles = batch["start_frame"]

    # styles comes in [-1, 1]
    # we want it in [0, 1]
    styles = (styles + 1.0) / 2
    styles = einops.rearrange(styles, "b h w c -> b c h w").cuda()

    control = batch["hed_intermediate_frame"]
    control = einops.rearrange(control, "b h w c -> b c h w").cuda()

    B, C, H, W = control.shape

    c_control = torch.cat([control, styles], axis=1)
    uc_control = torch.cat(
        [torch.zeros_like(control), torch.zeros_like(styles)], axis=1
    )

    c_prompt = model.get_learned_conditioning(
        ["a professional, detailed, high-quality image"] * B
    )
    uc_prompt = model.get_learned_conditioning([""] * B)

    cond = {
        "c_concat": [c_control],
        "c_crossattn": [c_prompt],
    }
    un_cond = {
        "c_concat": [uc_control],
        "c_crossattn": [uc_prompt],
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

    target = batch["intermediate_frame"]
    target = (target + 1.0) * 127.5
    target = target.clip(0, 255).type(torch.uint8)

    for j in range(B):
        Image.fromarray(x_samples[j]).save(f"samples_concat/x/img_{i}-{j}.png")
        Image.fromarray(target[j].cpu().numpy()).save(
            f"samples_concat/s/img_{i}-{j}.png"
        )
