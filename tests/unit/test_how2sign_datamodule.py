# import sys

# sys.path.insert(0, "/Users/mcciupek/Documents/42/AI/Hand2Text")

from src.datamodules.how2sign_datamodule import How2SignDataModule

if __name__ == "__main__":
    batch_size = 2

    datamodule = How2SignDataModule(batch_size=batch_size)
    datamodule.prepare_data()

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test

    datamodule.setup()

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))

    x, y = batch

    print(x)
    print(y)

    print(type(x))
    print(type(x[0]))

    assert len(x) == batch_size
    assert len(y) == batch_size
