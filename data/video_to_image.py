import json
import os
from typing import Dict, List, Literal, Tuple

import cv2

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = f"{ROOT_DIR}/data/H2T/WLASL_v0.3.json"

FrameData = Tuple[np.ndarray]


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


def get_frame_from_video(video_path: str, frame_subdir: str) -> List[FrameData]:
    """Returns array of frames for given frame_subdir.

    Args:
        video_path (str): Path for the video
        frame_subdir (str): Path where to store videos' frames

    Returns:
        List[FrameData]: List of frames with their datas
    """
    video_frames = []
    if not os.path.exists(frame_subdir):
        os.makedirs(frame_subdir)
        vid = cv2.VideoCapture(video_path)
        current_frame = 0

        while True:
            success, frame = vid.read()
            if not success:
                break
            elif current_frame % 25 == 0:
                full_filepath = f"{frame_subdir}/frame-{current_frame}.jpg"
                cv2.imwrite(full_filepath, frame)
                video_frames.append((frame,))
            current_frame += 1

        vid.release()
    else:
        frames_files = os.listdir(frame_subdir)
        for file in frames_files:
            frame = cv2.imread(f"{frame_subdir}/{file}")
            video_frames.append((frame,))

    cv2.destroyAllWindows()
    return video_frames


def main():
    SUB_DIR = f"{ROOT_DIR}/data/H2T"
    FRAMES_DIR = f"{SUB_DIR}/frames"
    RAW_VIDEOS_PATH = f"{SUB_DIR}/raw_videos"
    all_file = os.listdir(RAW_VIDEOS_PATH)
    len_all_file = len(all_file)
    labels = load_labels()
    data: List[FrameData] = []

    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
    for i, file in enumerate(all_file, 1):
        frame_subdir = f"{FRAMES_DIR}/{file.split('.')[0]}"
        print("Extract frames from %s: %.2f%%" % (file, i * 100 / len_all_file))
        frames = get_frame_from_video(f"{RAW_VIDEOS_PATH}/{file}", frame_subdir)
        data.extend(frames)


if __name__ == "__main__":
    main()
