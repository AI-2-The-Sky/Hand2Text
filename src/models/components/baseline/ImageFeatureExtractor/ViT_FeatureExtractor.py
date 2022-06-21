import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nptyping import Float32, NDArray, Number, Shape, UInt
from torch import nn
from transformers import ViTModel

# TorchFrames =


class ViT_FeatureExtractor(pl.LightningModule):
    def __init__(
        self,
        corpus: str = "/usr/share/dict/words",
    ):
        super().__init__()

        self.vocabulary_size = len(np.array(open(corpus).read().splitlines()))

        self.pretrained_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.pretrained_vit.eval()

        self.conv_1d_1 = torch.nn.Conv1d(
            in_channels=197,
            out_channels=64,
            kernel_size=3,
        )
        self.layer_1_relu = nn.ReLU()
        self.conv_1d_2 = torch.nn.Conv1d(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
        )
        self.layer_2_relu = nn.ReLU()

    def forward(
        self, x: NDArray[Shape["* batch, 224, 224, 3"], Float32]
    ) -> NDArray[Shape["* batch, * vocab size"], Float32]:

        outputs = self.pretrained_vit(pixel_values=x)
        vit_feat = outputs.last_hidden_state

        x = self.conv_1d_1(vit_feat)
        x = self.layer_1_relu(x)
        x = self.conv_1d_2(x)
        x = self.layer_2_relu(x)
        x = torch.squeeze(x)
        return x
