
import sys
sys.path.append('..')
import torch.nn as nn
from models.tensorized_components.multihead_attention import MultiHeadAttention as MHA
import torch
from einops import rearrange
from tensorized_layers.TLE import TLE
from tensorized_layers.TDLE import TDLE
from tensorized_layers.TP import TP

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Encoder(nn.Module):
    def __init__(self,
                input_size,
                patch_size,
                embed_dim,
                num_heads,
                mlp_dim,
                dropout=0.5,
                bias = True,
                out_embed = True,
                drop_path =0.1,
                ignore_modes=(0,1,2),
                tensor_method_mlp= ('tle','tle'),
                tensor_method='tle',
                tdle_level = 3,
                reduce_level = (0,0,0),
                ):
        super(Encoder, self).__init__()
        
        assert isinstance(tensor_method_mlp, tuple) and len(tensor_method_mlp) == 2, "tensor_method_mlp must be a tuple of length 2"
        assert all(m in {'tle', 'tdle', 'tp'} for m in tensor_method_mlp), f"Invalid methods in tensor_method_mlp: {tensor_method_mlp}"
        
        self.input_size = input_size
        self.patch_size = patch_size
        self.mlp_dim = tuple(x - y for x, y in zip(mlp_dim, reduce_level))
        self.embed_dim = tuple(x - y for x, y in zip(embed_dim, reduce_level))
        self.bias = bias
        self.ignore_modes = ignore_modes
                                
        self.tensor_input_size_layer1 =  (self.input_size[0], self.input_size[2]//self.patch_size + 1, self.input_size[3]//self.patch_size,
                                self.embed_dim[0], self.embed_dim[1], self.embed_dim[2])
        
        self.tensor_input_size_layer2 =  (self.input_size[0], self.input_size[2]//self.patch_size + 1, self.input_size[3]//self.patch_size,
                                self.mlp_dim[0], self.mlp_dim[1], self.mlp_dim[2])
        

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.attention = MHA(
                            input_size=input_size,
                            patch_size=patch_size,  
                            embed_dim= embed_dim,
                            num_heads=num_heads,
                            bias=bias,
                            out_embed= out_embed,
                            ignore_modes=ignore_modes,  
                            tensor_method=tensor_method, 
                            tdle_level=tdle_level,
                            reduce_level=reduce_level
                             )
        
        if tensor_method_mlp[0]=='tdle':
            layer1 = TDLE(input_size=self.tensor_input_size_layer1, rank=self.mlp_dim, ignore_modes= ignore_modes, bias=bias, r=tdle_level)
        elif tensor_method_mlp[0]=='tle':
            layer1 = TLE(input_size=self.tensor_input_size_layer1, rank=self.mlp_dim, ignore_modes= ignore_modes, bias=bias)
        elif tensor_method_mlp[0]=='tp':
            rank = self.embed_dim + self.mlp_dim
            output_size = tuple(x + y for x, y in zip(self.mlp_dim, reduce_level))
            layer1 = TP(input_size=self.tensor_input_size_layer1, output_size=output_size, rank=rank, ignore_modes= ignore_modes, bias=bias) 
        else:
            raise   ValueError(f"tensor method not defined {tensor_method_mlp[0]} or its handler is missing")
        
        if tensor_method_mlp[1]=='tdle':
            layer2 = TDLE(input_size=self.tensor_input_size_layer2, rank=self.embed_dim, ignore_modes= ignore_modes, bias=bias, r=tdle_level)
        elif tensor_method_mlp[1]=='tle':
            layer2 = TLE(input_size=self.tensor_input_size_layer2, rank=self.embed_dim, ignore_modes= ignore_modes, bias=bias)
        elif tensor_method_mlp[1]=='tp':
            rank = self.mlp_dim + self.embed_dim
            output_size = tuple(x + y for x, y in zip(self.embed_dim, reduce_level))
            layer2 = TP(input_size=self.tensor_input_size_layer2, output_size=output_size, rank=rank, ignore_modes= ignore_modes, bias=bias) 
        else:
            raise   ValueError(f"tensor method not defined {tensor_method_mlp[0]} or its handler is missing")
        
        self.mlp = nn.Sequential(
                        layer1,
                        nn.GELU(),
                        nn.Dropout(dropout), 
                        layer2,
                        nn.Dropout(dropout)
                    )
        
    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x