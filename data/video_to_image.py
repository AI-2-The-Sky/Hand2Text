import json
import os
from typing import Dict, List, Literal, Tuple

import cv2
from numpy import ndarray
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = f"{ROOT_DIR}/data/H2T/WLASL_v0.3.json"


class VideoMetadata:
    split_count = {"train": 0, "test": 0, "val": 0}

    def __init__(
        self, label: int, bbox: List[int], fps: int, split: Literal["train", "test", "val"]
    ):
        self.label = label
        self.bbox = bbox
        self.fps = fps
        self.split = split
        VideoMetadata.split_count[split] += 1


FrameMetaData = Tuple[ndarray, VideoMetadata]
FrameData = Tuple[ndarray, int]


def load_labels() -> Tuple[Dict[str, VideoMetadata], List[str]]:
    """Returns labels.

    Returns:
        Tuple[Dict[str, VideoMetadata], List[str]]:
            [0]: dict[video_name, metadata]
            [1]: list where label index correspond to a word
    """
    with open(JSON_PATH) as ipf:
        json_data = json.load(ipf)

    videos_labels: Dict[str, VideoMetadata] = {}
    words: List[str] = []
    label = -1
    for ent in json_data:
        word = ent["gloss"]
        label += 1
        words.append(word)

        for inst in ent["instances"]:
            videos_labels[inst["video_id"]] = VideoMetadata(
                label, inst["bbox"], inst["fps"], inst["split"]
            )
    return (videos_labels, words)


def frame_meta_to_label(frames: List[FrameMetaData]) -> List[FrameData]:
    """Convert list of frames with metadata to only labels
    Args:
        frames (List[FrameMetaData]): Frames to convert
    Returns:
        List[FrameData]: Converted frames
    """
    return list(map(lambda frame: (frame[0], frame[1].label), frames))


def get_frame_from_video(
    video_path: str, frame_subdir: str, label: VideoMetadata, download: bool, transform=None
) -> List[FrameMetaData]:
    """Returns array of frames for given frame_subdir.

    Args:
        video_path (str): Path for the video
        frame_subdir (str): Path where to store videos' frames

    Returns:
        List[FrameMetaData]: List of frames with their datas
    """
    video_frames = []
    if download and not os.path.exists(frame_subdir):
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
                video_frames.append((frame, label))
            current_frame += 1

        vid.release()
    else:
        frames_files = os.listdir(frame_subdir)
        for file in frames_files:
            frame = cv2.imread(f"{frame_subdir}/{file}")
            if transform:
                pil_image = Image.fromarray(frame)
                frame = transform(pil_image)
            video_frames.append((frame, label))

    cv2.destroyAllWindows()
    return video_frames


def load_dataset(download: bool = False, transform=None) -> Tuple[List[FrameMetaData], List[str]]:
    """Returns the dataset with metadata, and the word-labels as a list.

    Args:
        download (bool, optional): Has to cut missing frames. Defaults to False.

    Returns:
        Tuple[List[FrameMetaData], List[str]]:
            [0]: Dataset
            [1]: List of words, dataset's label being word's index
    """
    SUB_DIR = f"{ROOT_DIR}/data/H2T"
    FRAMES_DIR = f"{SUB_DIR}/frames"
    RAW_VIDEOS_PATH = f"{SUB_DIR}/raw_videos"
    all_file = os.listdir(RAW_VIDEOS_PATH)
    len_all_file = len(all_file)
    labels, words = load_labels()
    data: List[FrameMetaData] = []

    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)

    if os.path.exists(f"{SUB_DIR}/wlasl_words"):
        with open(f"{SUB_DIR}/wlasl_words", "w") as words_file:
            words_file.write("\n".join(words))

    if download:
        print("Downloading dataset...")
    else:
        print("Reframing videos...")

    for file in tqdm(all_file):
        video_name = file.split(".")[0]
        frame_subdir = f"{FRAMES_DIR}/{video_name}"

        if not (video_name in labels.keys()):
            continue

        frames = get_frame_from_video(
            f"{RAW_VIDEOS_PATH}/{file}", frame_subdir, labels[video_name], download, transform
        )

        data.extend(frames)
    return (data, words)
