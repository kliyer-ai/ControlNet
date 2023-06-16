from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from annotator.hed import HEDdetector

apply_hed = HEDdetector()

model = create_model("./models/cldm_v15_concat.yaml").cpu()
model.load_state_dict(
    load_state_dict(
        "./train_log/kin_hed_concat5/lightning_logs/version_10/checkpoints/epoch=0-step=101699.ckpt",
        location="cuda",
    )
)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(
    input_image,
    style_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale1,
    scale2,
    scale3,
    seed,
    eta,
    low_threshold,
    high_threshold,
):
    scales = [scale1, scale2, scale3]
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        style_image = resize_image(HWC3(style_image), image_resolution)

        H, W, C = img.shape

        detected_map = apply_hed(img)  # [h w]
        detected_map = HWC3(detected_map)  # [h w c] c = 3

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        # NOTE
        # old models use style im in [-1,1]
        # new ones in [0, 1]
        style = torch.from_numpy(style_image.copy()).float().cuda() / 255.0
        style = torch.stack([style for _ in range(num_samples)], dim=0)
        style = einops.rearrange(style, "b h w c -> b c h w").clone()

        print("control", control)
        print("style", style)

        zero_cond = torch.zeros_like(style)

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond1 = {
            "c_concat": [torch.cat([control, zero_cond], axis=1)],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        cond2 = {
            "c_concat": [torch.cat([zero_cond, style], axis=1)],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        cond3 = {
            "c_concat": [torch.cat([control, style], axis=1)],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": [
                torch.cat([zero_cond, zero_cond], axis=1)
                if guess_mode
                else torch.cat([control, zero_cond], axis=1)
            ],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            [cond1, cond2, cond3],
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scales,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="numpy", label="canny")
            style_image = gr.Image(source="upload", type="numpy", label="style")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(
                    label="Images", minimum=1, maximum=12, value=1, step=1
                )
                image_resolution = gr.Slider(
                    label="Image Resolution",
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                )
                strength = gr.Slider(
                    label="Control Strength",
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.01,
                )
                guess_mode = gr.Checkbox(label="Guess Mode", value=False)
                low_threshold = gr.Slider(
                    label="Canny low threshold",
                    minimum=1,
                    maximum=255,
                    value=100,
                    step=1,
                )
                high_threshold = gr.Slider(
                    label="Canny high threshold",
                    minimum=1,
                    maximum=255,
                    value=200,
                    step=1,
                )
                ddim_steps = gr.Slider(
                    label="Steps", minimum=1, maximum=100, value=20, step=1
                )
                scale1 = gr.Slider(
                    label="Guidance Scale Control Only",
                    minimum=0.0,
                    maximum=30.0,
                    value=9.0,
                    step=0.1,
                )
                scale2 = gr.Slider(
                    label="Guidance Scale Style Only",
                    minimum=0.0,
                    maximum=30.0,
                    value=9.0,
                    step=0.1,
                )
                scale3 = gr.Slider(
                    label="Guidance Scale Both",
                    minimum=0.0,
                    maximum=30.0,
                    value=9.0,
                    step=0.1,
                )
                seed = gr.Slider(
                    label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True
                )
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(
                    label="Added Prompt", value="best quality, extremely detailed"
                )
                n_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                )
        with gr.Column():
            result_gallery = gr.Gallery(
                label="Output", show_label=False, elem_id="gallery"
            ).style(grid=2, height="auto")
    ips = [
        input_image,
        style_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        ddim_steps,
        guess_mode,
        strength,
        scale1,
        scale2,
        scale3,
        seed,
        eta,
        low_threshold,
        high_threshold,
    ]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name="0.0.0.0")
