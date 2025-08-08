
import sys
sys.path.append('..')
import torch.nn as nn
from models.basic_components.multihead_attention import MultiHeadAttention as MHA
import torch

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
                 bias=True,
                 out_embed=True,
                 drop_path =0.1):
        super(Encoder, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        
        self.attention = MHA(
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            out_embed=out_embed,
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout), 
        )

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x