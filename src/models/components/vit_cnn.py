from typing import List

import torch
from transformers import ViTFeatureExtractor

from src.models.components.simple_cnn import SimpleCNNModel


class ViTCNNModel(SimpleCNNModel):
    def __init__(
        self,
        channels: int = 3,
        width: int = 1280,
        height: int = 720,
        kernel_size: int = 5,
        corpus: str = "/usr/share/dict/words",
        do_resize: bool = True,
        do_normalize: bool = True,
        image_mean: List[int] = None,
        image_std: List[int] = None,
    ):
        super().__init__(channels, width, height, kernel_size, corpus)

        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ViTFeatureExtractor(
            do_resize=self.do_resize,
            size=(self.width, self.height),
            do_normalize=self.do_normalize,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        x = super().forward(x)
        return x
