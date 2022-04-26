import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms


class SimpleCNNModel(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        width: int = 28,
        height: int = 28,
        corpus: str = "/usr/share/dict/words",
    ):
        super().__init__()

        self.channels = channels
        self.width = width
        self.height = height
        self.corpus = np.array(open(corpus).read().splitlines())
        self.n = 10

        self.conv1 = nn.Conv2d(self.channels, self.channels * 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.channels * 2, self.channels * 4, 5)

        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def to_str(self, pred: torch.Tensor) -> list:
        indices = pred.numpy()
        return self.corpus[indices]
