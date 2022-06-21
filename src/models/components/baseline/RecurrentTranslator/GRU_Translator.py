from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from nptyping import Float32, NDArray, Number, Shape, UInt
from torch import nn


class GRU_Translator(pl.LightningModule):
    def __init__(
        self,
        H_input_size: int = 764,
        H_output_size: int = 100,
        num_layers: int = 1,
        dropout: int = 0,
        corpus: str = "/usr/share/dict/words",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocabulary_size = len(np.array(open(corpus).read().splitlines()))
        self.layer_gru = nn.GRU(
            input_size=self.hparams.H_input_size,
            hidden_size=self.hparams.H_output_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )

        self.layer_1_dense = nn.Linear(self.hparams.H_output_size, self.hparams.H_output_size)
        self.layer_1_relu = nn.ReLU()
        self.layer_2_dense = nn.Linear(self.hparams.H_output_size, self.vocabulary_size)
        self.layer_2_relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layer_gru(X)
        X = self.layer_1_dense(X)
        X = self.layer_1_relu(X)
        X = self.layer_2_dense(X)
        X = self.layer_2_relu(X)
        X = self.softmax(X)
        return X
