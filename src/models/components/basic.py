import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class BasicModel(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        width: int = 28,
        height: int = 28,
        corpus: str = "/usr/share/dict/words",
    ):
        super().__init__()

        # Count number of available categories in corpus
        self.corpus = np.array(open(corpus).read().splitlines())
        self.n = 10  # len(corpus)

        # Layer 1
        self.fc_1 = nn.Linear(width * height * channels, self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Random prediction
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        return x

    def to_str(self, pred: torch.Tensor) -> list:
        # convert prediction (tensor of ints) to np.array of strings
        indices = pred.numpy()
        return self.corpus[indices]
