import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nptyping import Float32, NDArray, Number, Shape, UInt
from torch import nn
from transformers import ViTModel


class ViT_Conv1d_FeatureExtractor(pl.LightningModule):
    def __init__(
        self,
        nb_classes: int = 10,
        batch_size: int = 2,
        seq_size: int = 2,
        out_features: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.seq_size = seq_size
        self.pretrained_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.pretrained_vit.eval()
        self.conv_1d_1 = torch.nn.Conv1d(
            in_channels=197,
            out_channels=self.hparams.out_features,
            kernel_size=768,
        )
        self.layer_1_relu = nn.ReLU()

    def vit_extract_features(self, x):
        # print("---VIT EXTRACT FEATURES---")
        with torch.no_grad():
            outputs = self.pretrained_vit(pixel_values=x)
            vit_feat = outputs.last_hidden_state
        return vit_feat

    def forward(
        self,
        vit_feat,
    ) -> NDArray[Shape["* batch, * vocab size"], Float32]:
        # print("---VIT FORWARD---")
        x = self.conv_1d_1(vit_feat)
        x = self.layer_1_relu(x)
        x = torch.squeeze(x, dim=2)
        return x
