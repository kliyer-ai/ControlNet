from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
from custom_dataset_cross import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch


# Configs
resume_path = './models/control_v15_cross_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


dataset = MyDataset('kin')



# print(dataset[0])

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15_cross.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model = model.cuda()
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

print(model)

dp = dataset[0]

dp['jpg'] = torch.Tensor([dp['jpg']])
dp['hint'] = torch.Tensor([dp['hint']])
dp['style'] = torch.Tensor([dp['style']])
dp['txt'] = [dp['txt']]

x, c = model.get_input(dp, 1)


# 768
# v = torch.ones((1,512,512,3))
# c = torch.ones((1,77,768))
o = model(x,c)
print(o)

# Misc
# dataset = MyDataset('kin')
# logger = ImageLogger(batch_frequency=logger_freq, name='kin_cross_2')
# wandb_logger = WandbLogger(name='kin_cross_2', project="ControlNet")
# tbl = TensorBoardLogger(save_dir='ControlNet', name='kin_cross_2')

# dataloader = DataLoader(dataset, num_workers=64, batch_size=batch_size, shuffle=True)

# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], logger=[wandb_logger, tbl])



# Train!
# trainer.fit(model, dataloader)
