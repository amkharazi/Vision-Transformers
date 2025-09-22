import sys

sys.path.append("..")
import torch
import torch.nn as nn
from einops import rearrange

from tensorized_layers.TLE import TLE
from tensorized_layers.TDLE import TDLE
from tensorized_layers.TP import TP


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_size,
        patch_size,
        embed_dim,
        bias=True,
        device="cuda",
        ignore_modes=(0, 1, 2),
        tensor_layer_type="tle",
        tdle_level=3,
    ):
        super(PatchEmbedding, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.bias = bias
        self.device = device
        self.ignore_modes = ignore_modes

        self.tensor_input_size = (
            self.input_size[0],
            self.input_size[2] // self.patch_size,
            self.input_size[3] // self.patch_size,
            self.patch_size,
            self.patch_size,
            self.input_size[1],
        )

        if tensor_layer_type == "tle":
            self.tensor_layer = TLE(
                input_size=self.tensor_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
            )
        elif tensor_layer_type == "tp":
            self.tensor_layer = TP(
                input_size=self.tensor_input_size,
                output=self.embed_dim,
                rank=self.embed_dim * 2,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
            )
        else:
            self.tensor_layer = TDLE(
                input_size=self.tensor_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
                r=tdle_level,
            )

    def forward(self, x):
        x = rearrange(
            x,
            "b c (p1 h) (p2 w) -> b p1 p2 h w c",
            h=self.patch_size,
            w=self.patch_size,
        )
        x = self.tensor_layer(x)
        return x
