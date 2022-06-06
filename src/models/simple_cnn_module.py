import os
import re
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils.BlueScore import BleuScore

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))
VOCABULARY = "vocabulary"


class SimpleCNNModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        # corpus: str = "/usr/share/dict/words",
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        print("WE ARE USING OUR SimpleCNNModule")
        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # self.corpus = np.array(open(corpus).read().splitlines())
        # self.n = len(self.corpus)

        # metrics
        self.train_acc = BleuScore()
        self.val_acc = BleuScore()
        self.test_acc = BleuScore()

        self.val_acc_best = MaxMetric()

        self.corpus = np.array(open(f"{ROOT_DIR}/data/How2Sign/{VOCABULARY}").read().splitlines())

    def forward(self, x: torch.Tensor):
        # print(f"Forward")
        return self.net(x)

    def step(self, batch: Any):
        # print(f"STEP")
        x, y = batch
        # print(f"{x.size() = }")
        logits = self.forward(x)
        # print(f"{logits.size() = }")
        # print(f"{y.size() = }")
        # 		   [0,   1,  2]
        # logits = [2]
        # y = [1]
        # logits = [.2, .3, .5]
        # y = 	   [0,   1,  0]

        nb_examples, nb_classes = logits.size()
        vec_y = y.long()

        # 		   [0,   1,  2]
        # logits = [2]
        # y = [1]
        # logits = [.2, .3, .5]
        # y = 	   [0,   1,  0]

        # One hot: 1, 3
        # y = [1, 0, 1, 1, 2]
        # y = [
        # 		[0,   1,  0]

        # y = [0] -> [1,   0,  0]
        # y = [1] -> [0,   1,  0]
        # y = [2] -> [0,   0,  2]
        # print(f"{y.size() = }")
        # print(f"{vec_y.size() = }")
        # print(f"{vec_y = }")
        # y_2d = F.one_hot(vec_y.long(), num_classes=nb_classes)
        # print(f"{y_2d.size() = }")
        # print(f"{y_2d.type() = }")
        # print(f"{logits.size() = }")
        # print(f"{logits.type() = }")
        # import sys
        loss = self.criterion(logits, vec_y)
        preds = torch.argmax(logits, dim=1)
        # sys.exit(0)
        return loss, preds, vec_y

    def training_step(self, batch: Any, batch_idx: int):
        # print(f"TRAINING STEP")
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
                        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
