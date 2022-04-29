import json
import os
from typing import Dict, List, Literal

import cv2

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = f"{ROOT_DIR}/data/H2T/WLASL_v0.3.json"


class VideoMetadata:
    split_count = {"train": 0, "test": 0, "val": 0}

    def __init__(
        self, label: str, bbox: List[int], fps: int, split: Literal["train", "test", "val"]
    ):
        self.label = label
        self.bbox = bbox
        self.fps = fps
        self.split = split
        VideoMetadata.split_count[split] += 1


def load_labels() -> Dict[str, VideoMetadata]:
    with open(JSON_PATH) as ipf:
        json_data = json.load(ipf)

    videos_labels: Dict[str, VideoMetadata] = {}
    for ent in json_data:
        gloss = ent["gloss"]
        for inst in ent["instances"]:
            videos_labels[inst["video_id"]] = VideoMetadata(
                gloss, inst["bbox"], inst["fps"], inst["split"]
            )
    return videos_labels


def get_frame_from_video(filepath: str, frame_subdir: str):
    vid = cv2.VideoCapture(filepath)
    current_frame = 0

    while True:
        success, frame = vid.read()
        if not success:
            break
        elif current_frame % 25 == 0:
            full_filepath = f"{frame_subdir}/frame-{current_frame}.jpg"
            cv2.imwrite(full_filepath, frame)
        current_frame += 1

    vid.release()
    cv2.destroyAllWindows()
    return


def main():
    SUB_DIR = f"{ROOT_DIR}/data/H2T"
    FRAMES_DIR = f"{SUB_DIR}/frames"
    RAW_VIDEOS_PATH = f"{SUB_DIR}/raw_videos"
    all_file = os.listdir(RAW_VIDEOS_PATH)
    len_all_file = len(all_file)
    labels = load_labels()

    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
    for i, file in enumerate(all_file, 1):
        frame_subdir = f"{FRAMES_DIR}/{file.split('.')[0]}"
        print("Extract frames from %s: %.2f%%" % (file, i * 100 / len_all_file))
        if os.path.exists(frame_subdir):
            continue
        else:
            os.makedirs(frame_subdir)
        get_frame_from_video(f"{RAW_VIDEOS_PATH}/{file}", frame_subdir)


if __name__ == "__main__":
    main()
