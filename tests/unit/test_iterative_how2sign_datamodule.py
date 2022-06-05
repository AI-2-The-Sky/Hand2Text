# import os
# import sys

# ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.insert(0, ROOT_DIR)

from src.datamodules.iterative_how2sign_datamodule import How2SignDataModule

if __name__ == "__main__":
    batch_size = 2

    datamodule = How2SignDataModule(batch_size=batch_size)

    assert (
        not datamodule.train_dataset and not datamodule.val_dataset and not datamodule.test_dataset
    )

    datamodule.setup()

    assert datamodule.train_dataset and datamodule.val_dataset and datamodule.test_dataset

    # assert datamodule.train_dataloader()
    # assert datamodule.val_dataloader()
    # assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))

    x, y = batch

    print(x)
    print(y)

    print(type(x))
    print(type(x[0]))

    print(type(y))
    print(type(y[0]))

    # assert len(x) == batch_size
    # assert len(y) == batch_size
