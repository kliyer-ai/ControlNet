import glob
from torch.utils.data import DataLoader
from custom_dataset_cross import MyDataset
import cv2
from share import *
import torchvision


dataset = MyDataset("kin_hed2")
dataloader = DataLoader(dataset, num_workers=1, batch_size=1, shuffle=True)


def get_key(name: str):
    name = name.split(".")[1]
    name = name.split("_")[-1]
    name = name[3:]
    return int(name)


def get_sequence():
    for batch in dataloader:
        meta_batch = batch["meta"]["file_name"]
        meta = meta_batch[0]
        meta = meta.split("/")[1]
        meta = meta.split("_")[:-1]
        meta = "_".join(meta)
        print(meta)
        styles = glob.glob("./data/kin_hed2/jpg/" + meta + "*")
        styles.sort(key=get_key)
        structures = glob.glob("./data/kin_hed2/hint/" + meta + "*")
        structures.sort(key=get_key)
        print(styles)
        print(structures)
        return styles, structures


def get_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32) / 255.0
    return img


def main():
    import einops
    import numpy as np
    from ldm.modules.encoders.modules import FrozenClipImageEmbedder
    import torch.nn.functional as F
    import config
    import gradio as gr
    import random
    import torch
    from PIL import Image
    from pytorch_lightning import seed_everything
    from annotator.util import resize_image, HWC3
    from cldm.model import create_model, load_state_dict
    from cldm.ddim_hacked import DDIMSampler

    num_samples = 1
    ddim_steps = 20
    strength = 1
    eta = 0
    scale = 2

    p = "./train_log/kin_hed_2/lightning_logs/version_0/checkpoints/epoch=1-step=119930.ckpt"
    # p = './train_log/kin_hed_cross_3/lightning_logs/version_0/checkpoints/epoch=4-step=412223.ckpt'

    model = create_model("./models/cldm_v15_org.yaml").cpu()
    model.load_state_dict(load_state_dict(p, location="cuda"))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    frozenClipImageEmbedder = FrozenClipImageEmbedder()
    frozenClipImageEmbedder = frozenClipImageEmbedder.cuda()

    seed = 42
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    styles, structures = get_sequence()
    styles = np.array([get_img(style) for style in styles])
    styles = styles.copy().astype(np.float32) / 127.5 - 1.0

    structures = np.array([get_img(structure) for structure in structures])
    structures = structures.astype(np.float32) / 255.0

    B, H, W, C = structures.shape

    control = torch.from_numpy(structures.copy()).float().cuda()
    control = einops.rearrange(control, "b h w c -> b c h w")

    style = frozenClipImageEmbedder(styles)
    c_style = style.last_hidden_state
    c_embed = style.pooler_output

    uc_style = c_style  # torch.zeros_like(c_style) #
    uc_embed = c_embed

    cond = {
        "c_concat": [control],
        "c_crossattn": [
            model.get_learned_conditioning(
                ["a professional, detailed, high-quality image"] * B
            )
        ],
        "c_style": [c_style],
        "c_embed": [c_embed],
    }
    un_cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([""] * B)],
        "c_style": [uc_style],
        "c_embed": [uc_embed],
    }
    shape = (4, H // 8, W // 8)

    model.control_scales = [
        strength
    ] * 13  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
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

    print(samples.shape)
    x_samples = model.decode_first_stage(samples)
    x_samples = einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5

    grid = torchvision.utils.make_grid(x_samples, nrow=1)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.cpu().numpy().clip(0, 255).astype(np.uint8)

    x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)

    print(grid.shape)
    Image.fromarray(x_samples[0]).save("test_img1.png")
    Image.fromarray(grid).save("test_grid.png")


if __name__ == "__main__":
    main()
