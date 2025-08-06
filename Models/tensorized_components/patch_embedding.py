
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from einops import rearrange

from tensorized_layers.TLE import TLE
from tensorized_layers.TP import TP
from tensorized_layers.TDLE import TDLE

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 input_size,
                 patch_size,
                 embed_dim,
                 bias = True,
                 ignore_modes = (0,1,2),
                 tensor_method='tle',
                 tdle_level = 3,
                 reduce_level = (0,0,0),
                 ):
        super(PatchEmbedding, self).__init__()
        
        assert isinstance(embed_dim, tuple), f"embed_dim must be a tuple, but got {type(embed_dim).__name__}"
        assert tensor_method in {'tle', 'tdle', 'tp'}, f"Invalid tensor_method '{tensor_method}'. Must be one of: tle, tdle, tp."
        assert len(input_size) == 4, f"input_size must be a tuple of (B, C, H, W), but got {input_size}"
        assert input_size[2] % patch_size == 0 and input_size[3] % patch_size == 0, f"Input height ({input_size[2]}) and width ({input_size[3]}) must be divisible by patch_size ({patch_size})."
        assert len(embed_dim) == 3, f"embed_dim must be a 3D tuple (e.g., (d1, d2, d3)) for tensor factorization, got {embed_dim}"
        assert len(reduce_level) == 3, f"reduce_level must be a 3D tuple to match embed_dim, got {reduce_level}"     
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = tuple(x - y for x, y in zip(embed_dim, reduce_level))
        
        self.bias = bias
        self.ignore_modes = ignore_modes

        self.tensor_input_size =  (self.input_size[0], self.input_size[2]//self.patch_size, self.input_size[3]//self.patch_size,
                                self.patch_size, self.patch_size, self.input_size[1])
        if tensor_method=='tdle':                        
            self.tensor_layer = TDLE(
                            input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias, 
                            r= tdle_level
                            )
        elif tensor_method=='tle':
            self.tensor_layer = TLE(input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias,
                            )
        elif tensor_method=='tp':
            rank = tuple(x - y for x, y in zip(self.input_size[-3:], reduce_level)) + self.embed_dim
            output_size = tuple(x + y for x, y in zip(self.embed_dim, reduce_level))
            self.tensor_layer = TP(input_size=self.input_size,
                                   output_size=output_size,
                                   rank=rank,
                                   ignore_modes=self.ignore_modes,
                                   bias=self.bias
                                   )
        else:
            raise   ValueError(f"tensor method not defined {tensor_method} or its handler is missing")

    def forward(self, x):
        x = rearrange(x, 
                        'b c (p1 h) (p2 w) -> b p1 p2 c h w',
                        h=self.patch_size, w=self.patch_size)
        x = self.tensor_layer(x)
        return x