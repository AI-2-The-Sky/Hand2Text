import itertools
import os
import random
import re
from os.path import exists

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import IterableDataset
from torchvision.datasets.folder import make_dataset

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
VID_DIR = "raw_videos/train"
LAB_FILE = "how2sign_realigned.csv"
VOCABULARY = "vocabulary"


def get_vocabulary(labels, save: bool = False, path: str = None):

    # Init vocabulary:
    #   - <EOW>: End of word
    #   - <EOS>: End of sentence
    vocabulary = ["", "<EOW>", "<EOS>"]

    # Split sentences
    for sentence in labels:
        for word in re.sub(r"[^\w]", " ", sentence[0]).split():
            if word.lower() not in vocabulary:
                vocabulary.append(word.lower())

    if save is True and path is not None:
        file = open(path, "w")
        for word in vocabulary:
            file.writelines(word + "\n")
        file.close()

    return vocabulary


def get_samples(source: str, target: str, extensions=(".mp4", ".avi")):
    # Read source sentences
    data = pd.read_csv(source, delimiter="\t")

    samples = data.iloc[:, 3].values.tolist()
    labels = data.iloc[:, -1:].values.tolist()

    get_vocabulary(labels, save=True, path=target)

    return samples, labels


def encode(labels, path_to_voc):
    splited_sentence = [word.lower() for word in re.sub("[^\w]", " ", labels[0]).split()]
    vocabulary = np.array(open(path_to_voc).read().splitlines())
    encoded_sentence = [np.where(vocabulary == word)[0][0] for word in splited_sentence]
    return torch.LongTensor(encoded_sentence)


class How2Sign(IterableDataset):
    """
    Tuto: https://pytorch.org/vision/main/auto_examples/plot_video_api.html
    """

    def __init__(
        self,
        root: str = f"{ROOT_DIR}/data/How2Sign",
        videos_dir: str = VID_DIR,
        labels_file: str = LAB_FILE,
        vocabulary: str = VOCABULARY,
        epoch_size=None,
        transform=None,
        n_frames=16,
    ):
        super(How2Sign).__init__()

        # Paths
        self.root = root
        self.video_dir = f"{self.root}/{videos_dir}"
        self.labels_file = f"{self.root}/{labels_file}"
        self.vocab_file = f"{self.root}/{vocabulary}"

        # Get samples and labels
        self.samples, self.labels = get_samples(source=self.labels_file, target=self.vocab_file)

        self.check_files()

        self.len = len(self.labels)

        if epoch_size is None:
            epoch_size = self.len
        self.epoch_size = epoch_size

        self.n_frames = n_frames
        self.transform = transform

    def __iter__(self):
        for _ in range(self.epoch_size):
            # Get random sample
            i = random.randint(0, self.len - 1)
            video, label = self.samples[i], self.labels[i]

            # Read video
            video_frames, audio, metadata = torchvision.io.read_video(
                f"{self.video_dir}/{video}.mp4", pts_unit="sec"
            )
            # video_frames /= 255
            video_frames = torch.div(video_frames, 255)
            # video_frames = video_frames.double()
            # video_frames = video_frames.to(dtype=torch.float64)

            # Select n frames
            idx = np.linspace(0, video_frames.shape[0] - 1, self.n_frames, dtype=int)
            video_frames = video_frames[idx, :, :, :]

            # Transform
            if self.transform:
                video_frames = self.transform(video_frames)

            # Permute tensor dimension
            video_frames = video_frames.permute(0, 3, 1, 2)

            # Encode sentences
            y = encode(label, self.vocab_file)

            # Yield
            output = {"frames": video_frames, "label": y}

            yield output
            # yield video_frames, y

    def check_files(self):
        i = 0
        while i < len(self.samples):
            if not exists(f"{self.video_dir}/{self.samples[i]}.mp4"):
                self.samples.remove(self.samples[i])
                self.labels.remove(self.labels[i])
            else:
                i += 1
