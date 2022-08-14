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
    ) -> None:
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.load_from_checkpoint(ckpt_path)#.to(device=self.device)
        self.model.eval()
        self.output_format = output_format
        
    
    def pre_process(self, input_path: str, x_shape) -> torch.tensor:
        """Load and preprocess X
        """
        print(f"load video from {input_path}")
        frames, _, _ = read_video(input_path, pts_unit='sec')
        # THWC -> TCHW
        freq = 25
        N = frames.shape[0] // freq
        idx = np.arange(0, N * freq, freq)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=x_shape),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        X = transform(frames[idx, :, :, :])
        frames = frames.permute(0, 3, 1, 2).float()
        X_min = torch.min(X, dim=2, keepdim=True)
        print(X_min)
        X_max = torch.max(X, dim=2, keepdim=True)
        X = (X - X_min) / (X_max - X_min)
        X = X.unsqueeze(dim=0)
        print(X)
        return X
    
    def predict(self, video_path: str, x_shape = (224, 224)):
        X = self.pre_process(video_path, x_shape)
        print("X:", X.shape)
        print("preprocess OK")
        with torch.no_grad():
            y = self.model.forward(X)
        print("forward OK")
        print("y:", y.shape)
        # y.cpu().detach().numpy()
        # print("y:", y.shape)
        # print("to_numpy OK")
        # np.save(f'{self.output_path}/{self.folder}/pred.npy', y)