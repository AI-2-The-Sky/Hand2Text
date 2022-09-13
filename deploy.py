import os
import sys

import gradio as gr
import numpy as np
from torchvision.io import read_video
from torchvision.transforms import transforms

from infer import Infer
from src.models.BaseSquareConv1dModule import BaseSquareConv1dModule
from src.models.components.baseline.BaseSquareNetConv1d import BaseSquareNetConv1d

# module_path = os.path.abspath(os.path.join('..'))

# if module_path not in sys.path:
#     sys.path.insert(0, module_path)


def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)


def test(video):
    # dataset = SignedDataset(video, "Hello World!", 16)
    print(video)
    # net = BaseSquareNetConv1d(batch_size=1,
    #                             seq_size=25,
    #                             nb_classes=1999,
    #                             h_in=10,
    #                             k_features=64
    #                          )
    model = BaseSquareConv1dModule
    infer = Infer(
        model=model,
        freq=25,
        ckpt_path="logs/experiments/runs/basesquareconv1d_training_full/2022-09-02_12-07-08/checkpoints/last.ckpt",
    )
    y = infer.predict(video_path=video)
    return " ".join(y)


if __name__ == "__main__":

    # model = BaseSquareConv1dModule.load_from_checkpoint(
    #     checkpoint_path="/Users/mcciupek/Documents/42/AI/Hand2Text/logs/experiments/runs/basesquareconv1d_test/2022-08-10_21-31-11/checkpoints/last.ckpt",
    # )
    # freeze_all_layers_(model)
    # model.net.eval()

    demo = gr.Interface(fn=test, inputs=gr.inputs.Video(source="webcam"), outputs="text")
    demo.launch()
