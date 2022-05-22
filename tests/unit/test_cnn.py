import random

from src.datamodules.how2sign_datamodule import How2SignDataModule
from src.models.components.simple_cnn import SimpleCNNModel
from src.models.simple_cnn_module import SimpleCNNModule

# import sys

# sys.path.insert(0, "/Users/mcciupek/Documents/42/AI/Hand2Text")


if __name__ == "__main__":

    random.seed(42)

    batch_size = 2

    datamodule = How2SignDataModule(batch_size=batch_size)

    datamodule.prepare_data()
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))

    x, y = batch

    print(y)

    net = SimpleCNNModel(
        corpus="/Users/mcciupek/Documents/42/AI/Hand2Text/data/How2Sign/vocabulary"
    )
    module = SimpleCNNModule(net)

    loss, preds, targets = module.step(batch)

    print(loss)
    print(preds)
    print(targets)

    acc = module.train_acc(preds, targets)
    print(acc)

    true = [x[0] for x in targets]
    print(true)

    acc = module.train_acc(true, targets)

    assert acc == 1
