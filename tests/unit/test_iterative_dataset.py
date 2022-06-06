# import os
# import sys

# ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))
# sys.path.insert(0, ROOT_DIR)

import torchvision
from torch.utils.data import DataLoader

from data.IterativeHow2Sign import How2Sign

if __name__ == "__main__":

    dataset = How2Sign()
    loader = DataLoader(dataset, batch_size=10)

    data = {"video": [], "tensorsize": []}

    print(dataset.root)
    print(dataset.epoch_size)

    for batch in loader:
        x, y = batch
        for i in range(len(y)):
            data["video"].append(x[i])
            data["tensorsize"].append(x[i].size())

    print(x)
    print(y)
