from cgi import parse_multipart
from math import ceil
from typing import List, Optional, Tuple

import numpy as np
import torch
from nptyping import Float32, LongDouble, NDArray, Number, Shape, UInt
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset, random_split
from torchvision.transforms import transforms
from transformers import ViTFeatureExtractor

from data.video_to_image import frame_meta_to_label, load_dataset

# FrameMetaData = Tuple[ndarray, VideoMetadata]


RawFrameData = NDArray[Shape["3, 320, 240"], Float32]
RawVocabData = int


TorchFrames = NDArray[Shape["* batch, 224, 224, 3"], Float32]
TorchLabels = NDArray[Shape["* batch"], LongDouble]
# ProcessedDataset =


class SignedDataset(Dataset):
    def __init__(self, data, corpus):
        self.data = data
        self.subset_size = 32
        self.size = len(data)
        self.loaded_subset = -1
        self.vocabulary_size = len(np.array(open(corpus).read().splitlines()))
        model_name_or_path = "google/vit-base-patch16-224-in21k"
        self.image_preproccesing = ViTFeatureExtractor.from_pretrained(model_name_or_path)

    def one_hot_y(
        self, Y: NDArray[Shape["* nb examples"], UInt]
    ) -> NDArray[Shape["* nb examples, * vocabulary size"], UInt]:
        (nb_examples,) = Y.shape

        one_hot_Y = torch.zeros(nb_examples, self.vocabulary_size)
        for m in range(nb_examples):
            one_hot_Y[m, int(Y[m])] = 1
        return one_hot_Y

    def _load_subset(
        self, subset: Tuple[List[RawFrameData], List[RawVocabData]]
    ) -> Tuple[TorchFrames, TorchLabels]:
        multi_video_frames = []
        multi_video_signes = []
        for frames, signes in subset:
            if not len(frames):
                continue
            f = self.image_preproccesing
            pp_frames = [f(frame, return_tensors="pt")["pixel_values"] for frame in frames]
            t_frames = torch.cat(pp_frames, dim=0)
            multi_video_frames.append(t_frames)
            t_signes = torch.tensor(signes, dtype=torch.long)
            multi_video_signes.append(t_signes)
        t_video_frames = torch.cat(multi_video_frames, dim=0)
        t_video_signes = torch.cat(multi_video_signes, dim=0)
        return t_video_frames, t_video_signes

    def _get_subset(self, subset_nb: int) -> Tuple[TorchFrames, TorchLabels]:
        start = self.subset_size * subset_nb
        end = min(self.size, self.subset_size * (subset_nb + 1))
        subset = self._load_subset(self.data[start:end])
        return subset

    def _get_subset_nb(self, i: int) -> int:
        subset_nb = int(i / self.subset_size)
        return subset_nb

    def __getitem__(self, i: int) -> Tuple[TorchFrames, TorchLabels]:
        subset_nb = self._get_subset_nb(i)
        if subset_nb != self.loaded_subset:
            self.subset = self._get_subset(subset_nb)
            self.loaded_subset = subset_nb
        X, Y = self.subset
        X_i = X[i % self.subset_size]
        Y_i = Y[i % self.subset_size]
        return X_i, Y_i

    def __len__(self) -> int:
        return self.size


class ViTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        width=320,
        height=240,
        corpus: str = "/usr/share/dict/words",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

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
        X, y = dataset[0]
        print(f"{' ' * 4 * 2}X: {X.shape}")
        print(f"{' ' * 4 * 2}Y: {y.shape}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        Set variables: `self.data_train`, `self.data_val`, `self.data_test`. This method is
        called by lightning when doing `trainer.fit()` and `trainer.test()`, so be careful
        not to execute the random split twice! The `stage` can be used to differentiate
        whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # Load dataset
        dataset, words = load_dataset(transform=self.transforms)

        self.dataset = SignedDataset(dataset[: self.hparams.batch_size * 4], self.hparams.corpus)

        self.words = words

        test_size = int(len(self.dataset) * 0.25)
        val_size = int(len(self.dataset) * 0.25)
        train_size = len(self.dataset) - (test_size + val_size)

        self._print_dataset_shape("dataset", self.dataset)
        self.data_train, self.data_test, self.data_val = random_split(
            self.dataset,
            [train_size, test_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        self._print_dataset_shape("dataset_train", self.data_train)
        self._print_dataset_shape("dataset_val", self.data_val)
        self._print_dataset_shape("dataset_test", self.data_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
