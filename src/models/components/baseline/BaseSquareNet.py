import imp
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from nptyping import Float32, NDArray, Number, Shape, UInt

# from src.models.components.baseline.ImageFeatureExtractor.ResNet_FeatureExtractor import (
# 	ResNet_FeatureExtractor,from src.models.components.baseline.ImageFeatureExtractor.ResNet_FeatureExtractor import (
# 	ResNet_FeatureExtractor,
# )
# )
from src.models.components.baseline.ImageFeatureExtractor.ViT_FeatureExtractor import (
    ViT_FeatureExtractor,
)
from src.models.components.baseline.RecurrentTranslator.GRU_Translator import GRU_Translator


class BaseSquareNet(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 1,
        seq_size: int = 1,
        nb_classes: int = 10,
        h_in: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.nb_seq_sizebatch = seq_size
        # self.image_feature_extractr = ViT_FeatureExtractor(
        # 	nb_classes=nb_classes,
        # 	batch_size=batch_size,
        # 	seq_size=seq_size
        # )
        self.recurrent_translator = GRU_Translator(
            nb_classes=nb_classes, H_input_size=h_in, num_layers=1, dropout=0
        )

    def forward(
        self, x: NDArray[Shape["* batch, 224, 224, 3"], Float32]
    ) -> NDArray[Shape["* batch, * vocab size"], Float32]:
        # print(f"{x.shape=}")
        x = self.recurrent_translator(x)
        return x
