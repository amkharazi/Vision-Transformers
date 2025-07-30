from TLE import TLE
import torch.nn as nn

class TDLE(nn.Module):
    def __init__(self, input_size, rank, ignore_modes=(0,), bias=True, r=3):
        super(TDLE, self).__init__()
        self.layers = nn.ModuleList([
            TLE(input_size, rank, ignore_modes, bias) for _ in range(r)
        ])
    
    def forward(self, x):
        return sum(layer(x) for layer in self.layers)
