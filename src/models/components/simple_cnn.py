import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms


def get_conv_shape(shape, kernel_size):
    return (shape[0] * 2, (shape[1] - (kernel_size - 1)), (shape[2] - (kernel_size - 1)))


def get_pool_shape(shape):
    return (shape[0], shape[1] / 2, shape[2] / 2)


def dense_arg(channels, width, height, kernel_size):
    x = (channels, width, height)
    x = get_conv_shape(x, kernel_size)
    x = get_pool_shape(x)
    x = get_conv_shape(x, kernel_size)
    x = get_pool_shape(x)
    return int(x[0] * x[1] * x[2])


class SimpleCNNModel(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        width: int = 1280,
        height: int = 720,
        kernel_size: int = 5,
        corpus: str = "/usr/share/dict/words",
    ):
        super().__init__()

        self.channels = channels
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.corpus = np.array(open(corpus).read().splitlines())
        self.n = len(self.corpus)

        self.conv1 = nn.Conv2d(self.channels, self.channels * 2, kernel_size=self.kernel_size)
        self.norm_1 = nn.BatchNorm2d(self.channels * 2)

        self.conv2 = nn.Conv2d(self.channels * 2, self.channels * 4, kernel_size=self.kernel_size)
        self.norm_2 = nn.BatchNorm2d(self.channels * 4)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(
            dense_arg(self.channels, self.width, self.height, self.kernel_size), 120
        )
        # self.fc1 = nn.Linear(52668, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Block 1
        x = F.relu(self.conv1(x))
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.pool(x)

        # Block2
        x = F.relu(self.conv2(x))
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # FCN
        # Layer 1
        x = self.drop(x)
        x = F.relu(self.fc1(x))

        # Layer 2
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)

        # Layer 3
        x = self.fc3(x)
        return x

    def to_str(self, pred: torch.Tensor) -> list:
        indices = pred.numpy()
        return self.corpus[indices]
