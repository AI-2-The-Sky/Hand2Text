from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from nptyping import Float32, NDArray, Number, Shape, UInt

from src.models.components.baseline.ImageFeatureExtractor.ResNet_FeatureExtractor import (
    ResNet_FeatureExtractor,
)
from src.models.components.baseline.ImageFeatureExtractor.ViT_FeatureExtractor import (
    ViT_FeatureExtractor,
)
from src.models.components.baseline.RecurrentTranslator.GRU_Translator import GRU_Translator


class BaseSquareNet(pl.LightningModule):
    def __init__(
        self,
        corpus: str = "/usr/share/dict/words",
        sequence_size: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocabulary_size = len(np.array(open(corpus).read().splitlines()))
        # self.image_feature_extractr = ViT_FeatureExtractor(corpus)
        self.image_feature_extractr = ResNet_FeatureExtractor(corpus)
        self.recurrent_translator = GRU_Translator(
            H_input_size=764,
            H_output_size=100,
            num_layers=1,
            dropout=0,
            corpus=corpus,
        )

    def forward(
        self, x: NDArray[Shape["* batch, 224, 224, 3"], Float32]
    ) -> NDArray[Shape["* batch, * vocab size"], Float32]:
        # x_seq = []
        # for i in range(self.hparams.sequence_size):
        #     print(f"{x.shape = }")
        #     b, f = x.shape
        #     x = x.view((b, 1, f))
        #     print(f"{x.shape = }")
        #     x_seq.append(x)
        # x_seq = torch.cat(x_seq, dim=1)
        # print(f"In: {x.shape = }")
        x = self.image_feature_extractr(x)
        # print(f"Vit: {x.shape = }")/
        b, f = x.shape
        x_seq = x.view(1, b, f)
        # print(f"View: {x_seq.shape = }")
        x = self.recurrent_translator(x_seq)
        # print(f"gru: {x.shape = }")
        return x
