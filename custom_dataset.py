import json
import cv2
import numpy as np

from torch.utils.data import Dataset

PATH = './data/mine/'

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./data/char/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        hint_filename = item['hint']
        target_filename = item['target']
        prompt = item['prompt']

        hint = cv2.imread(PATH + hint_filename)
        source = cv2.imread(PATH + source_filename)
        target = cv2.imread(PATH + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        hint = cv2.cvtColor(hint, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        hint = hint.astype(np.float32) / 255.0

        # Normalize source images to [-1, 1].
        source = (source.astype(np.float32) / 127.5) - 1.0

        # start should probably be in [-1, 1] as that's the standard range for images

        # My Test condition
        hint = np.concatenate([source,hint], axis=-1)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=hint)

