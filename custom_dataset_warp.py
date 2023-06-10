import json
import cv2
import numpy as np

from torch.utils.data import Dataset

from annotator.hed import HEDdetector

apply_hed = HEDdetector()

# https://stackoverflow.com/questions/44535091/convection-of-an-image-using-optical-flow


class MyDataset(Dataset):
    def __init__(self, name):
        self.path = "./data/" + name + "/"

        self.data = []
        with open(self.path + "data.json", "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        hint_filename = item["hint"]
        target_filename = item["target"]
        prompt = item["prompt"]

        hint_org = cv2.imread(self.path + hint_filename)
        source_org = cv2.imread(self.path + source_filename)
        target_org = cv2.imread(self.path + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        hint = cv2.cvtColor(hint_org, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target_org, cv2.COLOR_BGR2RGB)
        source = cv2.cvtColor(source_org, cv2.COLOR_BGR2RGB)

        target_gray = cv2.cvtColor(target_org, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(source_org, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            source_gray, target_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        hed_t0 = apply_hed(source)

        # COLOR_BGR2GRAY

        # Normalize hint images to [0, 1] as it's a black and white edge map.
        hint = hint.astype(np.float32) / 255.0

        # Learning: this should also just be normalized to [0, 1]
        # Will be rescaled in the model to [-1, 1]
        # source = source.astype(np.float32) / 255.0

        # My Test condition
        # hint = np.concatenate([source,hint], axis=-1)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # Normalize source images to [-1, 1] which will serve as the style.
        # DONT DO THIS BECAUSE HUGGING FACE AUTO PROCESSOR EXPECTS [0, 255]
        # source = (source.astype(np.float32) / 127.5) - 1.0

        # style does not need to get normalized as it happens in the model later
        return dict(
            jpg=target,
            txt="a professional, detailed, high-quality image",
            hint=hint,
            style=source,
            meta={"file_name": source_filename},
        )
