from typing import Any, List

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics import functional as tm_F
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.auroc import AUROC

# from DraftOptimizer.models.components.simple_dense_net import SimpleDenseNet


class CategorySingle_PredictionSequence_LitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
                                    - Computations (init).
                                    - Train loop (training_step)
                                    - Validation loop (validation_step)
                                    - Test loop (test_step)
                                    - Optimizers (configure_optimizers)

    Read the docs:
                                    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        corpus: str = "/usr/share/dict/words",
        nb_classes: int = 2,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.vocabulary_size = len(np.array(open(corpus).read().splitlines()))

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion = torch.nn.BCELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # self.pred_types = ["vict", "gold", "time"]

        self.auc = {
            "train": AUROC(num_classes=self.vocabulary_size),
            "val": AUROC(num_classes=self.vocabulary_size),
            "test": AUROC(num_classes=self.vocabulary_size),
        }
        self.acc = {
            "train": Accuracy(mdmc_average="global"),
            "val": Accuracy(mdmc_average="global"),
            "test": Accuracy(mdmc_average="global"),
        }

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_auc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        # print(f"{x.shape = }")
        # print(f"{y.shape = }")

        # y = y.view(1, -1)
        # # print(f"{y.shape = }")
        logits = self.forward(x)
        # logits = torch.moveaxis(logits, 1, 2)
        # print(f"{logits.shape = }")
        loss = self.criterion(logits, y)
        return loss, logits, y

    def log_all_seq(self, loss, preds, targets, split="train"):
        for i in range(preds.shape[2]):
            pred = preds[:, :, i]
            target = targets[:, i]
            acc = tm_F.accuracy(pred, target)
            self.log(
                f"{split}/acc/seq_{i}",
                acc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            auc = tm_F.auroc(pred, target, num_classes=self.vocabulary_size)
            self.log(
                f"{split}/auc/seq_{i}",
                auc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

    def log_all(self, loss, preds, targets, split="train"):
        self.log_all_seq(loss, preds, targets, split)
        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.acc[split].update(preds, targets)
        ret = self.auc[split](preds, targets)
        self.log(f"{split}/auc/all", ret, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.log_all(loss, preds, targets, split="train")

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        split = "train"
        ret = self.acc[split].compute()
        self.log(f"{split}/acc/all", ret, on_step=False, on_epoch=True, prog_bar=True)
        self.acc[split].reset()
        self.auc[split].reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.log_all(loss, preds, targets, split="val")

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        split = "val"
        # Accuracy
        ret = self.acc[split].compute()
        self.log(f"{split}/acc/all", ret, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc_best.update(ret)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.acc[split].reset()
        # AUROC
        ret = self.auc[split].compute()
        self.val_auc_best.update(ret)
        self.log("val/auc_best", self.val_auc_best.compute(), on_epoch=True, prog_bar=True)
        self.auc[split].reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.log_all(loss, preds, targets, split="test")

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        split = "test"
        ret = self.acc[split].compute()
        ret = self.auc[split].compute()
        self.acc[split].reset()
        self.auc[split].reset()

    def on_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
                                        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(recurse=True),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
