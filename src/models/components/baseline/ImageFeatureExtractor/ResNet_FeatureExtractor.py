import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nptyping import Float32, NDArray, Number, Shape, UInt
from torch import nn
from transformers import DetrForObjectDetection

# TorchFrames =


class ResNet_FeatureExtractor(pl.LightningModule):
    def __init__(
        self,
        corpus: str = "/usr/share/dict/words",
    ):
        super().__init__()

        self.vocabulary_size = len(np.array(open(corpus).read().splitlines()))

        self.pretrained_resnet = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.pretrained_resnet.eval()

        self.conv_1d_1 = torch.nn.Conv1d(
            in_channels=100,
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
        self.layer_1_dense = nn.Linear(252, 764)
        self.layer_3_relu = nn.ReLU()

    def forward(
        self, x: NDArray[Shape["* batch, 224, 224, 3"], Float32]
    ) -> NDArray[Shape["* batch, * vocab size"], Float32]:

        outputs = self.pretrained_resnet(pixel_values=x)
        resnet_feat = outputs.last_hidden_state

        print(f"{resnet_feat.shape = }")
        x = self.conv_1d_1(resnet_feat)
        x = self.layer_1_relu(x)
        x = self.conv_1d_2(x)
        x = self.layer_2_relu(x)
        x = torch.squeeze(x)
        x = self.layer_1_dense(x)
        x = self.layer_3_relu(x)
        return x
