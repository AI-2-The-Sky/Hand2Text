import imp
from asyncio import open_unix_connection
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch

# from nptyping import Float32, NDArray, Number, Shape, UInt
from nptyping import Float32, NDArray, Number, UInt

from src.models.components.baseline.ImageFeatureExtractor.Dimensionality_Reductor import (
    Dimensionality_Reductor,
)
from src.models.components.baseline.ImageFeatureExtractor.ViT_Conv1d_FeatureExtractor import (
    ViT_Conv1d_FeatureExtractor,
)
from src.models.components.baseline.RecurrentTranslator.GRU_Translator import GRU_Translator


class BaseSquareNetConv1d(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 2,
        seq_size: int = 1,
        nb_classes: int = 10,
        h_in: int = 10,
        k_features: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vit = ViT_Conv1d_FeatureExtractor(
            nb_classes=self.hparams.nb_classes,
            batch_size=self.hparams.batch_size,
            seq_size=self.hparams.seq_size,
            out_features=self.hparams.k_features,
        )

        self.dimensionality_reductor = Dimensionality_Reductor(
            out_features=self.hparams.k_features
        )
        self.recurrent_translator = GRU_Translator(
            nb_classes=nb_classes, H_input_size=h_in, num_layers=1, dropout=0
        )

    # def forward(
    #     self, x: NDArray[Shape["* batch, * seq, 3, 224, 224"], Float32]
    # ) -> NDArray[Shape["* batch, * vocab size"], Float32]:
    def forward(self, x):

        # print(f"start fwd {x.shape =}")
        # start fwd x.shape =torch.Size([1, 2, 3, 224, 224])

        b, s, c, k, f = x.size()
        x = x.view(b * s, c, k, f)

        # print(f"view {x.shape =}")
        # view x.shape =torch.Size([2, 3, 224, 224])

        x = self.vit.vit_extract_features(x)

        # print(f"vit {x.shape =}")
        # vit x.shape =torch.Size([2, 197, 768])

        x = self.dimensionality_reductor(x)

        # print(f"dimensionality_reductor {x.shape =}")
        # dimensionality_reductor x.shape =torch.Size([2, 64])

        # b = self.hparams.batch_size
        s = self.hparams.seq_size
        f = self.hparams.k_features

        x = torch.flatten(x, start_dim=1)
        bs, _f = x.size()

        # assert f == _f
        # print(f'{b =}')
        # print(f'{s =}')
        # print(f'{bs =}')
        # assert b * s == bs

        x = x.view(b, s, f)
        x = self.recurrent_translator(x)
        return x
