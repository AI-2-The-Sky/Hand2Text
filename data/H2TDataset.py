import json
import os

from torch.utils.data import Dataset

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = f"{ROOT_DIR}/data/H2T/WLASL_v0.3.json"

# Doc: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class H2TDataset(Dataset):
    """Hand2Text dataset."""

    def __init__(self, json_path=JSON_PATH, root_dir=ROOT_DIR, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(JSON_PATH) as ipf:
            self.json_data = json.load(ipf)

        self.root_dir = root_dir
        self.transform = transform

    # def __len__(self):
    #     return len(self.landmarks_frame)

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()

    #     img_name = os.path.join(self.root_dir,
    #                             self.landmarks_frame.iloc[idx, 0])
    #     image = io.imread(img_name)
    #     landmarks = self.landmarks_frame.iloc[idx, 1:]
    #     landmarks = np.array([landmarks])
    #     landmarks = landmarks.astype('float').reshape(-1, 2)
    #     sample = {'image': image, 'landmarks': landmarks}

    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample
