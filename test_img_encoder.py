from ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenOpenCLIPEmbedder, FrozenCLIPEmbedder
import torch
import cv2
from PIL import Image
import numpy as np
import requests




# with torch.no_grad(), torch.cuda.amp.autocast():
imgModel = FrozenClipImageEmbedder()
# openImgModel = FrozenOpenCLIPEmbedder()
textModel = FrozenCLIPEmbedder()

print(imgModel)

# the problems are 
# 1) the sequence length is too long for the image
# 2) the embedding size does not match for the 1.5 model



if torch.cuda.is_available():
    print('cuda available')
    imgModel.cuda()
    textModel.cuda()
# # model.eval()

img = cv2.imread('./data/char/source/LFC6D.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = torch.from_numpy(hint)
# img = Image.open('./data/char/source/LFC6D.png').convert('RGB')

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# img = Image.open(requests.get(url, stream=True).raw)

test = textModel(['hello world how are you doing? Thanks!', 'Get the memory footprint of a model. This will return the memory footprint of the current model in bytes. Useful to benchmark the memory footprint of the current model and design some tests.'])

res = imgModel(img)
print(res.shape)