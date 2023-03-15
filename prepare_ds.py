import cv2
import glob
from annotator.canny import CannyDetector
from annotator.util import resize_image, HWC3
import json
from pathlib import Path
import torch
from PIL import Image

apply_canny = CannyDetector()

from lavis.models import load_model_and_preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN= 512
low_threshold = 100
high_threshold = 200
OUT_PATH = './data/kin/'
IN_PATH = '/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020/train/*/*'
GENERATE_PROMPT = True
SKIP = 10

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite( pathOut + "\\frame%d.png" % count, image)     # save frame as JPEG file
        count = count + 1

def extract_start_end_imgs(pathIn):
    vidcap = cv2.VideoCapture(pathIn)
    _, start = vidcap.read()
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(1*1000))
    _, end = vidcap.read()
    return start, end

def center_crop_img(img):
    y, x, _ = img.shape
    k = min(y,x)

    center_x = x / 2 - k / 2
    center_y = y / 2 - k / 2

    return resize_image(img[int(center_y):int(center_y + k) , int(center_x):int(center_x + k),:], MIN)

model, vis_processors = None, None
def generate_prompt(raw_image):
    global model
    global vis_processors
    if model is None or vis_processors is None:
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    # lavis wants a PIL image
    raw_image = Image.fromarray(raw_image, mode='RGB')
    
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    prompts = model.generate({"image": image})
    prompt = prompts[0]
    print(prompt)
    return prompt

def main():
    Path(OUT_PATH + "source").mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH + "target").mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH + "hint").mkdir(parents=True, exist_ok=True)
    # read video
    paths = glob.glob(IN_PATH)
    out_str = ''

    for path in paths[::SKIP]:
        vidcap = cv2.VideoCapture(path)
        success, start = vidcap.read()
        if not success: continue

        y, x, _ = start.shape
        if y < MIN or x < MIN: 
            print('skipping...')
            continue

        start = center_crop_img(start)
        base_name = path.split('/')[-1].split('.')[0]
        cv2.imwrite(OUT_PATH + 'source/' + base_name + '.png', start)
        print(base_name)

        prompt = ""
        if GENERATE_PROMPT:
            # preprocess the image
            raw_image = cv2.cvtColor(start, cv2.COLOR_BGR2RGB)
            prompt = generate_prompt(raw_image)
        

        for frame in [1,2,4]:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(frame*1000))
            success, end = vidcap.read()
            if not success: continue

            end = center_crop_img(end)

            detected_map = apply_canny(end, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            name = base_name + '_f' + str(frame)
            cv2.imwrite(OUT_PATH + 'target/' + name + '.png', end)
            cv2.imwrite(OUT_PATH + 'hint/' + name + '.png', detected_map)
            out_str += json.dumps({
                "source": 'source/' + base_name + '.png',
                "target": 'target/' + name + '.png',
                "hint": 'hint/' + name + '.png',
                "prompt": prompt
            }) + '\n'

    with open(OUT_PATH + 'data.json', 'w') as f:
        f.write(out_str)

  

if __name__ == '__main__':
    main()