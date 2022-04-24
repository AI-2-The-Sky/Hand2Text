import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from random_word import RandomWords
from torch import nn


class BasicModel:
    def __init__(
        self,
        channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        self.words = open("/usr/share/dict/words").read().splitlines()
        self.r = RandomWords()

    def forward(self, X: torch.Tensor):
        # Random prediction
        corpus = self.r.get_random_words(limit=X.shape[0])
        return corpus
