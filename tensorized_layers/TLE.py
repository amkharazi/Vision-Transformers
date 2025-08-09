import torch
import torch.nn as nn
from torch.nn import init
import math

class TLE(nn.Module):
    def __init__(self, input_size, rank, ignore_modes=(0,), bias=True):
        super(TLE, self).__init__()

        if isinstance(input_size, int):
            input_size = (input_size,)
        self.input_size = tuple(input_size)
        self.ignore_modes = set(ignore_modes)
        self.rank = rank if isinstance(rank, (tuple, list)) else (rank,)

        self.project_modes = [i for i in range(len(self.input_size)) if i not in self.ignore_modes]
        assert len(self.project_modes) == len(self.rank), "Rank must match number of projected modes."

        self.linears = nn.ModuleDict()
        for i, mode in enumerate(self.project_modes):
            in_dim = self.input_size[mode]
            out_dim = self.rank[i]
            self.linears[str(mode)] = nn.Linear(in_dim, out_dim, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.empty(*self.rank))
        else:
            self.register_parameter('bias', None)

        self.init_parameters()

    def init_parameters(self):
        for linear in self.linears.values():
            init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
        if self.bias is not None:
            first_mode = self.project_modes[0]
            fan_in = self.input_size[first_mode]
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        for mode in self.project_modes:
            linear = self.linears[str(mode)]

            perm = list(range(x.ndim))
            perm[mode], perm[-1] = perm[-1], perm[mode]
            x = x.permute(perm)

            orig_shape = x.shape
            x = x.reshape(-1, orig_shape[-1])
            x = linear(x)
            x = x.reshape(*orig_shape[:-1], x.shape[-1])

            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            x = x.permute(inv_perm)

        if self.bias is not None:
            x = x + self.bias
        return x
