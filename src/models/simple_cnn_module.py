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
        loss = 0
        sentence = np.empty((x.size()[0], x.size()[1]), dtype="<U256")
        for i in range(x.size()[1]):
            if i == len(y[0]):
                break
            logits = self.forward(x[:, i, :, :])
            loss += self.criterion(logits, y[:, i])
            preds = self.corpus[torch.argmax(logits, dim=1)]
            sentence[:, i] = preds
        ground_truth = [[x] for x in self.corpus[y]]
        return loss, sentence.tolist(), ground_truth

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
