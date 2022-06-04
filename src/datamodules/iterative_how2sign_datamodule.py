# Code obtained from https://github.com/ashleve/lightning-hydra-template

import math
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    IterableDataset,
    get_worker_info,
    random_split,
)
from torchvision.transforms import transforms

from data.IterativeHow2Sign import How2Sign


def custom_collate(data):
    inputs = [d["frames"] for d in data]
    labels = [d["label"] for d in data]
    inputs = pad_sequence(inputs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    return inputs, labels


# def custom_worker_init_fn(worker_id):
#     worker_info = get_worker_info()
#     dataset = worker_info.dataset  # the dataset copy in this worker process
#     overall_start = dataset.start
#     overall_end = dataset.end
#     # configure the dataset to only process the split workload
#     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
#     worker_id = worker_info.id
#     dataset.start = overall_start + worker_id * per_worker
#     dataset.end = min(dataset.start + per_worker, overall_end)


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
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = None  # transforms.Compose([transforms.ToTensor()])

        self.dataset: Optional[IterableDataset] = None

        self.train_dataset: Optional[IterableDataset] = None
        self.val_dataset: Optional[IterableDataset] = None
        self.test_dataset: Optional[IterableDataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`. This method is
        called by lightning when doing `trainer.fit()` and `trainer.test()`, so be careful
        not to execute the random split twice! The `stage` can be used to differentiate
        whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        # if not self.data_train and not self.data_val and not self.data_test:
        # if not self.train_dataset and not self.val_dataset and not self.test_dataset:
        if not self.dataset:
            self.train_dataset = How2Sign(transform=self.transforms, videos_dir="raw_videos/train")
            self.val_dataset = How2Sign(transform=self.transforms, videos_dir="raw_videos/val")
            self.test_dataset = How2Sign(transform=self.transforms, videos_dir="raw_videos/test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # shuffle=True,
            collate_fn=custom_collate,
            # worker_init_fn=custom_worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=custom_collate,
            # worker_init_fn=custom_worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=custom_collate,
            # worker_init_fn=custom_worker_init_fn,
        )
