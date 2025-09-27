import sys

sys.path.append("..")
import torch.nn as nn
from models.basic_components.multihead_attention import MultiHeadAttention as MHA


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        patch_size,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout=0.5,
        bias=True,
        out_embed=True,
        device="cuda",
        ignore_modes=None,
        Tensorized_mlp=False,
    ):
        super(Encoder, self).__init__()
        self.tensorized_mlp = Tensorized_mlp

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = MHA(
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            out_embed=out_embed,
            device=device,
            ignore_modes=ignore_modes,
        )
        if self.tensorized_mlp == False:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, embed_dim),
                nn.Dropout(dropout),
            )
        else:
            self.mlp = self.tensorized_mlp[0]
            self.norm2 = self.tensorized_mlp[1]

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
