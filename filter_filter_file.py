import os
import numpy as np
import cv2
import json

mode = "val"
filter_file = "/export/home/koktay/flow_diffusion/scripts/timestamps_validation.json"
data_path = "/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020"
videos_dir = os.path.join(data_path, mode)
annotations_dir = os.path.join(data_path, "annotations", f"{mode}.csv")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def main():
    with open(annotations_dir, "r") as f:
        annotations = f.read().splitlines()[1:]

        # Get the video labels
        labels = list()
        for annotation in annotations:
            label = annotation.split(",")
            labels.append(
                {
                    "video_path": os.path.join(
                        videos_dir,
                        label[0],
                        f"{label[1]}_{label[2].zfill(6)}_{label[3].zfill(6)}.mp4",
                    ),
                }
            )

    with open(filter_file, "r") as f:
        obj = json.load(f)
        f.close()
        indices = np.array(obj["id"])
        timestamps_start = obj["timestamps_start"]
        timestamps_end = obj["timestamps_end"]

    new = {"id": [], "timestamps_start": [], "timestamps_end": []}
    MIN = 512

    for idx, ts, te in zip(indices, timestamps_start, timestamps_end):
        if idx >= len(labels):
            print("continue, larger than list", idx, len(labels))
            continue

        path = labels[idx]["video_path"]
        vidcap = cv2.VideoCapture(path)
        success, start = vidcap.read()

        if not success:
            continue

        y, x, _ = start.shape
        if y < MIN or x < MIN:
            print("skipping...")
            continue

        print("adding", path)
        new["id"].append(idx)
        new["timestamps_start"].append(ts)
        new["timestamps_end"].append(te)

    with open(f"data_{mode}.json", "w") as fp:
        json.dump(new, fp, cls=NpEncoder)


if __name__ == "__main__":
    main()
