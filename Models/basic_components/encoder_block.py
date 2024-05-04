
import sys
sys.path.append('..')
import torch.nn as nn
from Models.basic_components.multihead_attention import MultiHeadAttention as MHA

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.5, out_embed = True):
        super(Encoder, self).__init__()
        self.attention = MHA(
                            embed_dim = embed_dim,
                            num_heads = num_heads,
                            out_embed= out_embed
                            )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_res = x
        x = self.norm(x + self.dropout(self.attention(x)))
        x = self.norm(x + self.dropout(self.mlp(x)))
        return x + x_res