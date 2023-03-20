from ldm.modules.encoders.modules import FrozenClipImageEmbedder
import torch

model = FrozenClipImageEmbedder()
if torch.cuda.is_available():
    model.cuda()
model.eval()