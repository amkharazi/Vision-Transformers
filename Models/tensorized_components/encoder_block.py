
import sys
sys.path.append('..')
import torch.nn as nn
from Models.tensorized_components.multihead_attention import MultiHeadAttention as MHA
from einops import rearrange

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.5, out_embed = True, Tensorized = True):
        super(Encoder, self).__init__()
        self.attention = MHA(embed_dim = embed_dim,
                             num_heads = num_heads,
                             out_embed = out_embed
                             )
        
        self.Tensorized = Tensorized
        if not self.Tensorized:
            feature_dim = embed_dim[0]*embed_dim[1]*embed_dim[2]
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, feature_dim),
                nn.Dropout(dropout)
            )
        else:
            self.mlp = nn.Sequential(
                
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_res = x
        x = self.norm1(x + self.dropout(self.attention(x)))
        shapes = x.shape
        x = rearrange(x, 'b p1 p2 h w c -> b (p1 p2) (h w c)')
        x = self.norm2(x + self.dropout(self.mlp(x)))
        x = rearrange(x, 'b (p1 p2) (h w c) -> b p1 p2 h w c', p1 = shapes[1], p2 = shapes[2], h = shapes[3], w = shapes[4], c = shapes[5])
        return x + x_res