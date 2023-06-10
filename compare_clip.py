from custom_dataset_cross import MyDataset

# from share import *
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
# from torch.utils.data import DataLoader
# from cldm.logger import ImageLogger
# from cldm.model import create_model, load_state_dict


# resume_path = './models/control-base.ckpt'

# batch_size = 4
# logger_freq = 300
# learning_rate = 1e-5
# sd_locked = True
# only_mid_control = False
# experiment_name = 'kin_hed_2'


# model = create_model('./models/cldm_v15_org.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
# model.learning_rate = learning_rate
# model.sd_locked = sd_locked
# model.only_mid_control = only_mid_control

# =================================
import einops
import numpy as np
from ldm.modules.encoders.modules import FrozenClipImageEmbedder

import torch.nn.functional as F


from share import *
import config

import gradio as gr
import random
import torch

from PIL import Image
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


p = './train_log/kin_hed_2/lightning_logs/version_0/checkpoints/epoch=1-step=119930.ckpt'
# p = './train_log/kin_hed_cross_3/lightning_logs/version_0/checkpoints/epoch=4-step=412223.ckpt'

model = create_model('./models/cldm_v15_org.yaml').cpu()
model.load_state_dict(load_state_dict(p, location='cuda'))
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

dataset = MyDataset('kin_hed2')
num_samples = 1
ddim_steps = 20
strength = 1
eta = 0
scale = 2

for batch in dataset:

    target = batch['jpg']
    H, W, C = target.shape
    target = (target * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    target = np.stack([target], axis=0) 

    control = torch.from_numpy(batch['hint'].copy()).float().cuda()
    control = torch.stack([control], dim=0) 
    control = einops.rearrange(control, 'b h w c -> b c h w')

    #  # get encoding here from style encoder
    style = frozenClipImageEmbedder([batch['style']])
    c_style = style.last_hidden_state
    c_embed = style.pooler_output  
    uc_style =  c_style # torch.zeros_like(c_style) #
    uc_embed = c_embed

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([batch['txt']])], "c_style": [c_style], "c_embed": [c_embed] }
    un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([''])], "c_style": [uc_style], "c_embed": [uc_embed]  }
    shape = (4, H // 8, W // 8)

    model.control_scales =  ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond)



    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    
   

    target_embed = frozenClipImageEmbedder(target)
    # target_embed = target_embed.pooler_output
    target_embed = target_embed.last_hidden_state

    Image.fromarray(x_samples[0]).save('test_img1.png')
    Image.fromarray(target[0]).save('test_img2.png')

    sample_embed = frozenClipImageEmbedder(x_samples)
    # sample_embed = sample_embed.pooler_output  
    sample_embed = sample_embed.last_hidden_state


    rmse = torch.sqrt(torch.mean((sample_embed - target_embed)**2))
    l1 = torch.mean(torch.abs(sample_embed - target_embed))

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    print(rmse)
    # print('first')
    # print(sample_embed)
    # print('second')
    # print(target_embed)
    # print(cos(sample_embed, target_embed))
    print(F.cosine_similarity(sample_embed, target_embed))
# validation_set = MyDataset('kin_hed_val')





# logger = ImageLogger(batch_frequency=logger_freq, name=experiment_name)
# # wandb_logger = WandbLogger(name='kin_hed_cross_2', project="ControlNet")
# # tbl = TensorBoardLogger(save_dir='ControlNet', name='kin_hed_cross_2')

# dataloader = DataLoader(dataset, num_workers=64, batch_size=batch_size, shuffle=True)
# validation_loader = DataLoader(validation_set, num_workers=64, batch_size=batch_size, shuffle=True)


# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger],
#                     limit_val_batches=1,
#                     val_check_interval=logger_freq,
#                     num_sanity_val_steps=2,
#                     default_root_dir='train_log/' + experiment_name
#                     ) #, logger=[wandb_logger, tbl])



# # Train!
# trainer.fit(model, dataloader, validation_loader)
