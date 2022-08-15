import torch
from torchvision.transforms import transforms
from torchvision.io import read_video
import numpy as np

class Infer():
    def __init__(
        self,
        model,
        freq: int = 0,
        ckpt_path: str = "", 
        output_format: str = "TCHW",
        seq_size: int = 2,
        corpus: str = "data/H2T/wlasl_words"
    ) -> None:
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.load_from_checkpoint(ckpt_path)#.to(device=self.device)
        self.model.eval()
        self.output_format = output_format
        self.freq = freq
        self.seq_size = seq_size
        with open(corpus, 'r') as f:
            self.corpus = f.read().split('\n')
        
    def pre_process(self, input_path: str, x_shape) -> torch.tensor:
        """Load and preprocess X
        """
        print(f"load video from {input_path}")
        # TODO: error management for short videos
        # (if T (nb frames) < seq_size)
        frames, _, _ = read_video(input_path, pts_unit='sec')
        N = frames.shape[0] // self.freq
        idx = np.arange(0, N * self.freq, self.freq)
        transform = transforms.Compose(
            [
                transforms.Resize(size=x_shape),
            ]
        )
        frames = frames.permute(0, 3, 1, 2).float() # THWC -> TCHW
        X = transform(frames[idx, :, :, :] / 255) # normalize
        X = X.unsqueeze(dim=0) # add batch dim
        print(f'{X.shape = }')
        X = torch.split(X, split_size_or_sections=self.seq_size, dim=1)
        print(f'{len(X) = }')
        if X[-1].shape != X[0].shape:
            X = X[:-1]
        print(f'{len(X) = }')
        X = torch.cat(X, dim=0)
        print(f'{X.shape = }')
        return X

    def post_process(self, y):
        y = torch.argmax(y, dim=2)
        # print(y)
        y = torch.flatten(y)
        y = [self.corpus[idx] for idx in y]
        return y
    
    def predict(self, video_path: str, x_shape = (224, 224)):
        X = self.pre_process(video_path, x_shape)
        print("X:", X.shape)
        print("preprocess OK")
        with torch.no_grad():
            y = self.model.forward(X)
        print("forward OK")
        print("y:", y.shape)
        y = self.post_process(y)
        return y
        # y.cpu().detach().numpy()
        # print("y:", y.shape)
        # print("to_numpy OK")
        # np.save(f'{self.output_path}/{self.folder}/pred.npy', y)