import sys

# module_path = os.path.abspath(os.path.join('..'))

# if module_path not in sys.path:
#     sys.path.insert(0, module_path)

import os
import gradio as gr
# from src.models.BaseSquareConv1dModule import BaseSquareConv1dModule
from torchvision.transforms import transforms
from torchvision.io import read_video

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def test(video):
    # dataset = SignedDataset(video, "Hello World!", 16)
    print(video)
    frames, audio, metadata = read_video(video, pts_unit='sec', output_format="TCHW")
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
        ]
    )
    X = transform(frames)
    y = "Hello World!"
    print(X.shape)
    # print(y.shape)
    return "Hello World!"



if __name__ == "__main__":

    # model = BaseSquareConv1dModule.load_from_checkpoint(
    #     checkpoint_path="/Users/mcciupek/Documents/42/AI/Hand2Text/logs/experiments/runs/basesquareconv1d_test/2022-08-10_21-31-11/checkpoints/last.ckpt",
    # )
    # freeze_all_layers_(model)
    # model.net.eval()

    demo = gr.Interface(fn=test, inputs=gr.inputs.Video(source="webcam"), outputs="text")
    demo.launch()
