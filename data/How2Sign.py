import os
import re
from os.path import exists

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
VID_DIR = "raw_videos"
LAB_FILE = "how2sign_realigned.csv"
VOCABULARY = "vocabulary"
FRAME_FREQ = 25

# Doc: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class How2Sign(Dataset):
    """How2Sign dataset."""

    def __init__(
        self,
        root_dir: str = ROOT_DIR,
        videos_dir: str = VID_DIR,
        labels_file: str = LAB_FILE,
        frame_frequency: int = FRAME_FREQ,
        transform=None,
        download: bool = False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = f"{root_dir}/data/How2Sign"

        self.video_dir = f"{self.data_dir}/{videos_dir}"
        LABELS_PATH = f"{self.data_dir}/{labels_file}"

        data = pd.read_csv(LABELS_PATH, delimiter="\t")
        self.x_files = data.iloc[:, 3].values.tolist()
        self.y = data.iloc[:, -1:].values.tolist()

        vocabulary = [""]
        for sentence in self.y:
            for word in re.sub(r"[^\w]", " ", sentence[0]).split():
                if word.lower() not in vocabulary:
                    vocabulary.append(word.lower())

        file = open(f"{self.data_dir}/{VOCABULARY}", "w")
        for word in vocabulary:
            file.writelines(word + "\n")
        file.close()

        for i, file in enumerate(self.x_files):
            if not exists(f"{self.video_dir}/{file}.mp4"):
                self.x_files.remove(file)
                self.y.remove(self.y[i])

        self.frame_freq = frame_frequency
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        video_frames = []

        vid = cv2.VideoCapture(f"{self.video_dir}/{self.x_files[idx]}.mp4")
        current_frame = 0

        while True:
            success, frame = vid.read()
            if not success:
                break
            elif current_frame % self.frame_freq == 0:
                frame = np.array(frame)
                if self.transform:
                    frame = self.transform(frame)
                video_frames.append(frame)
            current_frame += 1

        vid.release()

        video_frames = torch.stack(video_frames)

        sentence = [word.lower() for word in re.sub(r"[^\w]", " ", self.y[idx][0]).split()]
        vocabulary = np.array(open(f"{self.data_dir}/{VOCABULARY}").read().splitlines())

        words = [np.where(vocabulary == word)[0][0] for word in sentence]
        words = torch.LongTensor(words)

        # sample = {"frames": video_frames, "label": self.y[idx]}
        sample = {"frames": video_frames, "label": words}

        return sample
