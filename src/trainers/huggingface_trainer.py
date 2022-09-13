# from types import NoneType

import pytorch_lightning as pl
import transformers


class HuggingFace_Trainer(pl.Trainer):
    def __init__(
        self,
        output_dir,
        max_epochs: int = 10,
        gpus: int = 0,
        detect_anomaly: bool = True,
        track_grad_norm: int = 2,
        callbacks: str = None,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.detect_anomaly = detect_anomaly
        self.track_grad_norm = track_grad_norm
        self.callbacks = callbacks
        self.args = transformers.TrainingArguments(output_dir=output_dir, push_to_hub=True)
