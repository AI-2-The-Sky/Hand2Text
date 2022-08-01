import os
import sys

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from src.datamodules.hand2text_vit_datamodule import Hand2TextViTDataModule
from src.models.components.baseline.BaseSquareNetConv1d import BaseSquareNetConv1d

if __name__ == "__main__":

    batch_size = 3
    seq_size = 2

    datamodule = Hand2TextViTDataModule(batch_size=batch_size)

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test

    datamodule.setup()

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    # assert datamodule.train_dataloader()
    # assert datamodule.val_dataloader()
    # assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))

    x, y = batch

    print(f"{x.shape = }")
    print(f"{y.shape = }")

    model = BaseSquareNetConv1d(batch_size=batch_size, seq_size=seq_size, k_features=10)

    pred = model.forward(x)

    print(f"{pred.shape = }")
    print(f"{pred = }")

    # assert len(x) == batch_size
    # assert len(y) == batch_size
