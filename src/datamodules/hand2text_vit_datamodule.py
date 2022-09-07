from typing import Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from data.video_to_image import frame_meta_to_label, load_dataset
from src.models.components.baseline.ImageFeatureExtractor.ViT_Conv1d_FeatureExtractor import (
    ViT_Conv1d_FeatureExtractor,
)


class SignedDataset(Dataset):
    def __init__(self, X, Y, seq_size):
        self.X = X
        # [n_video, nb_frames, 3, 320, 240]
        self.Y = Y
        # [n_video, nb_signes, 1]
        # self.vit = vit
        self.seq_size = seq_size

    def __len__(self):
        return self.X.shape[0] // self.seq_size

    def __getitem__(self, i):
        # print(f'[{i}]: {self.X.shape=}, {self.Y.shape=}')
        # print(f'{self.Y =}')
        x_idx_a = self.seq_size * i
        x_idx_z = self.seq_size * (i + 1)
        # print(f'[{i}]: {x_idx_a=}, {x_idx_z=}')
        batch_X = self.X[x_idx_a:x_idx_z]
        batch_Y = self.Y[x_idx_a:x_idx_z]
        # print(batch_Y)
        return batch_X, batch_Y


class Hand2TextViTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        seq_size: int = 2,
        nb_classes: int = 10,
        out_features: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        width=224,
        height=224,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.batch_size = batch_size
        self.seq_size = seq_size

        self.feature_extractor = ViT_Conv1d_FeatureExtractor(
            nb_classes=self.hparams.nb_classes,
            batch_size=batch_size,
            seq_size=seq_size,
            out_features=self.hparams.out_features,
        )

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size=(width, height)),
                transforms.ToTensor(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # Cut video to images
        load_dataset(True)
        pass

    def _print_dataset_shape(self, dataset_name, dataset):
        m = len(dataset)
        print(f"{' ' * 4}{dataset_name} shape is:")
        print(f"{' ' * 4 * 2}m: {m}")
        for i in range(m):
            X, y = dataset[i]
            print(f"{' ' * 4 * 2}X[{i}]: {X.shape}")
            print(f"{' ' * 4 * 2}Y[{i}]: {y.shape}")
            print(" ")
            break

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`. This method is
        called by lightning when doing `trainer.fit()` and `trainer.test()`, so be careful
        not to execute the random split twice! The `stage` can be used to differentiate
        whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # Load dataset
        dataset, words = load_dataset(transform=self.transforms)
        print(f"AFTER LOADT DATASET : {len(dataset) = }")
        # dataset = [
        # 	[f_1 + f_2 + f_3], [l_1 + l_2 + l_3]
        # 	[f_1 + f_2 + f_3], [l_1 + l_2 + l_3]
        # 	[f_1 + f_2 + f_3], [l_1 + l_2 + l_3]
        # # ]
        multi_video_frames = []
        multi_video_signes = []
        for frames, signes in dataset:
            if not len(frames):
                continue
            t_frames = torch.stack(frames, dim=0)
            # print(f"{t_frames.shape = }")
            multi_video_frames.append(t_frames)
            # [s1, s2, s3]
            t_signes = torch.Tensor(signes)
            # [[s1], [s2], [s3]]
            # print(f"{t_signes.shape = }")
            multi_video_signes.append(t_signes)

        t_video_frames = torch.cat(multi_video_frames, dim=0)
        t_video_signes = torch.cat(multi_video_signes, dim=0)
        print(f"{t_video_frames.shape =}")
        print(f"{t_video_signes.shape =}")
        # t_video_features = self.feature_extractor.vit_extract_features(t_video_frames)
        print(f"{t_frames[0].shape =}")

        print(f"{self.seq_size =}")
        self.dataset = SignedDataset(
            t_video_frames, t_video_signes, self.seq_size
        )

        print(f"{len(self.dataset) =}")
        print(f"{self.dataset[0][0].shape =}")


        self.words = words

        test_size = int(len(self.dataset) * 0.2)
        val_size = int(len(self.dataset) * 0.2)
        train_size = len(self.dataset) - (test_size + val_size)

        self._print_dataset_shape("dataset", self.dataset)

        self.data_train, self.data_test, self.data_val = random_split(
            self.dataset,
            [train_size, test_size, val_size]
            # generator=torch.Generator().manual_seed(42),
        )
        self._print_dataset_shape("dataset_train", self.data_train)
        self._print_dataset_shape("dataset_val", self.data_val)
        self._print_dataset_shape("dataset_test", self.data_test)

    def train_dataloader(self) -> DataLoader:
        print("*** DATALOADER ***")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            # batch_size=self.hparams.batch_size * self.hparams.seq_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            # batch_size=self.hparams.batch_size * self.hparams.seq_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            # batch_size=self.hparams.batch_size * self.hparams.seq_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
