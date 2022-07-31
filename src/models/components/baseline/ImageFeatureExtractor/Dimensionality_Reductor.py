import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nptyping import Float32, NDArray, Number, Shape, UInt
from torch import nn


class Dimensionality_Reductor(pl.LightningModule):
    def __init__(
        self,
        out_features: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conv_1d_1 = torch.nn.Conv1d(
            in_channels=197,
            out_channels=self.hparams.out_features,
            kernel_size=768,
        )
        self.layer_1_relu = nn.ReLU()

    def forward(
        self,
        vit_feat,
    ) -> NDArray[Shape["* batch, * vocab size"], Float32]:
        
        x = self.conv_1d_1(vit_feat)
        x = self.layer_1_relu(x)
        x = torch.squeeze(x, dim=2)
        return x
