import sys
sys.path.append('..')

import torch
import torch.nn as nn
from tensorized_layers.TLE import TLE

class TP(nn.Module):
    def __init__(self, input_size, output_size, rank, ignore_modes=(0,), bias=True):
        super().__init__()

        self.input_size = tuple(input_size)
        self.output_size = tuple(output_size)
        self.ignore_modes = tuple(ignore_modes)
        self.rank = tuple(rank)

        self.n = len(self.input_size) - len(self.ignore_modes)
        self.m = len(self.output_size)

        assert len(self.rank) == self.n + self.m

        self.input_rank = self.rank[:self.n]
        self.output_rank = self.rank[self.n:]

        self.input_projector = TLE(self.input_size, self.input_rank, ignore_modes=self.ignore_modes, bias=False)
        self.core = nn.Parameter(torch.randn(*self.input_rank, *self.output_rank))
        core_shape = self.input_rank + self.output_rank
        self.core_projector = TLE(core_shape, self.output_size, ignore_modes=tuple(range(self.n)), bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(*self.output_size))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x_hat = self.input_projector(x)
        g_hat = self.core_projector(self.core)
        out = torch.tensordot(x_hat, g_hat, dims=self.n)
        if self.bias is not None:
            out += self.bias
        return out
