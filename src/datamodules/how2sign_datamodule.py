# Code obtained from https://github.com/ashleve/lightning-hydra-template

from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from data.How2Sign import How2Sign


def custom_collate(data):
    inputs = [d["frames"] for d in data]
    labels = [d["label"] for d in data]
    inputs = pad_sequence(inputs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    return inputs, labels


class How2SignDataModule(LightningDataModule):
    """Example of LightningDataModule for How2Sign dataset. A DataModule implements 5 key methods:

        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        How2Sign(download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`. This method is
        called by lightning when doing `trainer.fit()` and `trainer.test()`, so be careful
        not to execute the random split twice! The `stage` can be used to differentiate
        whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = How2Sign(transform=self.transforms)
            train_val_test_split = [
                int(x * len(dataset)) for x in self.hparams.train_val_test_split
            ]
            train_val_test_split[2] = len(dataset) - (
                train_val_test_split[0] + train_val_test_split[1]
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=custom_collate,
        )
