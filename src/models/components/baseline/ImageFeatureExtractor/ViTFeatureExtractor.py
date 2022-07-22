import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from nptyping import Float32, NDArray, Number, Shape, UInt
from torch import nn
from transformers import ViTModel

class ViTFeatureExtractor(pl.LightningModule):
	def __init__(
		self,
		nb_classes: int = 10,
		batch_size: int = 2,
		seq_size: int = 2,
	):
		super().__init__()

		self.pretrained_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
		self.pretrained_vit.eval()
		self.batch_size = batch_size
		self.seq_size = seq_size

	def vit_extract_features(self, x):
		# print("---VIT EXTRACT FEATURES---")
		with torch.no_grad():
			outputs = self.pretrained_vit(pixel_values=x)
			vit_feat = outputs.last_hidden_state
			vit_feat = torch.flatten(vit_feat, start_dim=1)
			b, f = vit_feat.size()
			vit_feat = torch.reshape(vit_feat, (self.batch_size, self.seq_size, f))
		return vit_feat
	
	def forward(
		self,
		x, 
	) -> NDArray[Shape["* batch, * vocab size"], Float32]:
		pass