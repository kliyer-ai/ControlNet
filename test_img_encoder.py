from ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenOpenClipImageEmbedder, FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import torch
import cv2
from PIL import Image
import numpy as np
import requests




# with torch.no_grad(), torch.cuda.amp.autocast():
imgModel = FrozenClipImageEmbedder()
# openImgModel = FrozenOpenClipImageEmbedder()
# textModel = FrozenCLIPEmbedder()
# openTextModel = FrozenOpenCLIPEmbedder()

# the problems are 
# 1) the sequence length is too long for the image
# 2) the embedding size does not match for the 1.5 model



if torch.cuda.is_available():
    print('cuda available')
    imgModel.cuda()
    # openImgModel.cuda()
    # textModel.cuda()
    # openTextModel.cuda()
# # model.eval()

img = cv2.imread('./data/char/source/LFC6D.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = Image.open('./data/char/source/LFC6D.png').convert('RGB')
imgs = np.stack([img, img, img, img], axis=0)
# imgs = [img, img, img, img]
print(imgs)
# img = torch.from_numpy(hint)
# img = Image.open('./data/char/source/LFC6D.png').convert('RGB')

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# img = Image.open(requests.get(url, stream=True).raw)

text = ['hello world how are you doing? Thanks!', 'Get the memory footprint of a model. This will return the memory footprint of the current model in bytes. Useful to benchmark the memory footprint of the current model and design some tests.']
# test1 = textModel(text)
# test2 = openTextModel(text)

res1 = imgModel(imgs)
# res2 = openImgModel(img)
print(res1.shape)
# print(test1.shape)
print('-----------------')
# print(res2.shape)
# print(test2.shape)