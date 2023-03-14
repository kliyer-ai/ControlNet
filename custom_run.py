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





def process(input_image, style_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, style_strength, scale, seed, eta, low_threshold, high_threshold):
    apply_canny = CannyDetector()

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    with torch.no_grad():
        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        t_enc = int(style_strength * ddim_steps)
        
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        style_image = resize_image(HWC3(style_image), image_resolution)
        style_image = torch.from_numpy(style_image.copy()).float().cuda() / 255.0
        style_image = 2. * style_image - 1
        style_image = torch.stack([style_image for _ in range(num_samples)], dim=0)
        style_image = einops.rearrange(style_image, 'b h w c -> b c h w').clone()
        style_latent = model.get_first_stage_encoding(model.encode_first_stage(style_image))  # move to latent space

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        # encode (scaled latent)
        z_enc = ddim_sampler.stochastic_encode(style_latent, torch.tensor([t_enc] * num_samples).cuda())
        # decode it
        samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                    unconditional_conditioning=un_cond, )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


def main():
    canny_image = cv2.imread('./test/me_up.png' )
    style_image = cv2.imread('./test/me_diagonal.png/' )
    print(canny_image)
    return
    prompt = ''
    num_samples = 1
    image_resolution = 512
    strength = 1 # 0 - 2
    style_strength = 0.5 # 0 - 1
    guess_mode = False
    low_threshold = 100
    high_threshold = 200
    ddim_steps = 20
    scale = 9
    seed = 42
    eta = 0
    a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
    n_prompt = gr.Textbox(label="Negative Prompt",
                            value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
       
    imgs = process(canny_image, style_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, style_strength, scale, seed, eta, low_threshold, high_threshold)

    print(len(imgs), imgs[1].shape)


if __name__ == '__main__':
    main()