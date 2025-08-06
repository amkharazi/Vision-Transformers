
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensorized_layers.TLE import TLE
from tensorized_layers.TDLE import TDLE
from tensorized_layers.TP import TP

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_size,
                 patch_size,
                 embed_dim,
                 num_heads,
                 bias = True,
                 out_embed = True,
                 ignore_modes = (0,1,2),
                 tensor_method='tle',
                 tdle_level = 3,
                 reduce_level = (0,0,0),
                 ):
        super(MultiHeadAttention, self).__init__()
        
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

        self.h1 = num_heads[0]
        self.h2 = num_heads[1]
        self.h3 = num_heads[2]
        
        self.scale = ((embed_dim[0]//self.h1)*
                      (embed_dim[1]//self.h2)*
                      (embed_dim[2]//self.h3))** -0.5
        
        self.out_embed = out_embed
        
        self.tensor_input_size =  (self.input_size[0], self.input_size[2]//self.patch_size + 1, self.input_size[3]//self.patch_size,
                                self.embed_dim[0], self.embed_dim[1], self.embed_dim[2],)

        
        if tensor_method=='tdle':  
            self.tensor_layer_Q = TDLE(
                            input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias, 
                            r= tdle_level
                            )
            self.tensor_layer_K = TDLE(
                            input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias, 
                            r= tdle_level
                            )
            self.tensor_layer_V = TDLE(
                            input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias, 
                            r= tdle_level
                            )
            if self.out_embed:
                self.tensor_layer_out = TDLE(
                            input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias, 
                            r= tdle_level
                            )
        elif tensor_method=='tle':
            self.tensor_layer_Q = TLE(input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias,
                            )
            self.tensor_layer_K = TLE(input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias,
                            )
            self.tensor_layer_V = TLE(input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias,
                            )
            if self.out_embed:
                self.tensor_layer_out = TLE(input_size=self.tensor_input_size,
                            rank=self.embed_dim,
                            ignore_modes=self.ignore_modes,
                            bias=self.bias,
                            )
        elif tensor_method=='tp':
            rank = tuple(x - y for x, y in zip(self.input_size[-3:], reduce_level)) + self.embed_dim
            output_size = tuple(x + y for x, y in zip(self.embed_dim, reduce_level))
            self.tensor_layer_Q = TP(input_size=self.input_size,
                                   output_size=output_size,
                                   rank=rank,
                                   ignore_modes=self.ignore_modes,
                                   bias=self.bias
                                   )
            self.tensor_layer_K = TP(input_size=self.input_size,
                                   output_size=output_size,
                                   rank=rank,
                                   ignore_modes=self.ignore_modes,
                                   bias=self.bias
                                   )
            self.tensor_layer_V = TP(input_size=self.input_size,
                                   output_size=output_size,
                                   rank=rank,
                                   ignore_modes=self.ignore_modes,
                                   bias=self.bias
                                   )
            if self.out_embed:
                self.tensor_layer_out = TP(input_size=self.input_size,
                                   output_size=output_size,
                                   rank=rank,
                                   ignore_modes=self.ignore_modes,
                                   bias=self.bias
                                   )
        else:
            raise   ValueError(f"tensor method not defined {tensor_method} or its handler is missing")

    def forward(self, x):
        q = self.tensor_layer_Q(x)
        k = self.tensor_layer_K(x)
        v = self.tensor_layer_V(x)

        Q = rearrange(q, 'b p1 p2 (x h1) (y h2) (z h3) -> b h1 h2 h3 p1 p2 x y z', h1 = self.h1, h2 = self.h2, h3 = self.h3)
        K = rearrange(k, 'b p1 p2 (x h1) (y h2) (z h3) -> b h1 h2 h3 p1 p2 x y z', h1 = self.h1, h2 = self.h2, h3 = self.h3)
        V = rearrange(v, 'b p1 p2 (x h1) (y h2) (z h3) -> b h1 h2 h3 p1 p2 x y z', h1 = self.h1, h2 = self.h2, h3 = self.h3)

        attn = torch.matmul(
                    rearrange(Q, 'b h1 h2 h3 p1 p2 x y z -> b h1 h2 h3 (p1 p2) (x y z)'),
                    rearrange(K, 'b h1 h2 h3 p1 p2 x y z -> b h1 h2 h3 (p1 p2) (x y z)').transpose(-1, -2)
                ) * self.scale
        attention = F.softmax(attn, dim=-1)
        
        x = torch.matmul(
            attention,
            rearrange(V, 'b h1 h2 h3 p1 p2 x y z -> b h1 h2 h3 (p1 p2) (x y z)')
        )
        
        x = rearrange(x, 'b h1 h2 h3 (p1 p2) (x y z)  -> b p1 p2 (x h1) (y h2) (z h3)',
                      h1 = self.h1,
                      h2 = self.h2,
                      h3 = self.h3,
                      p1 = V.shape[4],
                      p2 = V.shape[5],
                      x  = V.shape[6],
                      y  = V.shape[7],
                      z  = V.shape[8])

        if self.out_embed:
            x = self.tensor_layer_out(x)

        return x