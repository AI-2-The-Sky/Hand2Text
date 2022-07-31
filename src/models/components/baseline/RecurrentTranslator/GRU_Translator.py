from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from nptyping import Float32, NDArray, Number, Shape, UInt
from torch import nn

# class GRU_Translator(pl.LightningModule):
#     def __init__(
#         self,
#         nb_classes,
#         H_input_size: int = 151296,
#         H_output_size: int = 10,
#         num_layers: int = 1,
#         dropout: int = 0,
#     ):
#         # print("---GRU INIT---")
#         super().__init__()
#         self.save_hyperparameters()
#         self.vocabulary_size = nb_classes
#         self.layer_gru = nn.GRU(
#             input_size=self.hparams.H_input_size,
#             hidden_size=self.hparams.nb_classes,
#             num_layers=self.hparams.num_layers,
#             batch_first=True,
#             dropout=self.hparams.dropout,
#         )

#         self.layer_1_dense = nn.Linear(self.hparams.nb_classes, int(self.hparams.nb_classes / 2))
#         self.layer_2_dense = nn.Linear(int(self.hparams.nb_classes / 2), self.hparams.nb_classes)

#         self.layer_1_relu = nn.ReLU()
#         # self.layer_2_dense = nn.Linear(self.hparams.H_output_size, self.vocabulary_size)
#         self.layer_leaky_relu = nn.LeakyReLU()
#         self.softmax = nn.Softmax(dim=2)  # <-- i/o dim=2

#     def forward(self, X):
#         X, _ = self.layer_gru(X)
#         X = self.layer_1_dense(X)
#         X = self.layer_1_relu(X)
#         X = self.layer_2_dense(X)
#         X = self.softmax(X)
#         return X


class GRU_Translator(pl.LightningModule):
    def __init__(
        self,
        nb_classes,
        H_input_size: int = 151296,
        H_output_size: int = 10,
        num_layers: int = 1,
        dropout: int = 0,
    ):
        # print("---GRU INIT---")
        super().__init__()
        self.save_hyperparameters()
        self.vocabulary_size = nb_classes
        self.layer_gru = nn.GRU(
            input_size=self.hparams.H_input_size,
            hidden_size=self.hparams.nb_classes,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
        )

        self.layer_1_dense = nn.Linear(self.hparams.nb_classes, self.hparams.nb_classes)

        self.layer_1_relu = nn.ReLU()
        self.layer_leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=2)  # <-- i/o dim=2

    def forward(self, X):
        X, _ = self.layer_gru(X)
        X = self.layer_1_dense(X)
        X = self.softmax(X)
        return X
