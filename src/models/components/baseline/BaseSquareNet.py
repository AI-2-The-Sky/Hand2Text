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
from src.models.components.baseline.ImageFeatureExtractor.ViTFeatureExtractor import (
	ViTFeatureExtractor,
)

from src.models.components.baseline.RecurrentTranslator.GRUTranslator import GRUTranslator


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

		self.vocabulary_size = 1999
		self.batch_size = batch_size
		self.seq_size = seq_size
		self.nb_classes = nb_classes
		self.h_in = h_in

		# self.image_feature_extractor = ViTFeatureExtractor(nb_classes=nb_classes, batch_size=batch_size, seq_size=seq_size)
		self.recurrent_translator = GRUTranslator(
			nb_classes=self.nb_classes,
			H_input_size=self.h_in,
			H_output_size=100,
			num_layers=1,
			dropout=0,
		)

	def forward(
		self, x: NDArray[Shape["* batch, 224, 224, 3"], Float32]
	) -> NDArray[Shape["* batch, * seq, * vocab size"], Float32]:

		# x = self.image_feature_extractor.vit_extract_features(x)
		x = self.recurrent_translator(x)
		return x
