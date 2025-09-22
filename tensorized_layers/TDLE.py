import sys

sys.path.append("..")

import torch
import torch.nn as nn

from tensorized_layers.TLE import TLE


class TDLE(nn.Module):
    def __init__(
        self, input_size, rank, ignore_modes=(0,), bias=True, device="cuda", r=3
    ):
        super(TDLE, self).__init__()
        self.blocks = nn.ModuleList(
            [TLE(input_size, rank, ignore_modes, bias, device) for _ in range(r)]
        )

    def forward(self, x):
        return sum(block(x) for block in self.blocks)
