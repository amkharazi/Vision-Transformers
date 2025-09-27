import sys

sys.path.append("..")
import torch.nn as nn
from models.tensorized_components.multihead_attention import MultiHeadAttention as MHA
from einops import rearrange
from tensorized_layers.TLE import TLE
from tensorized_layers.TDLE import TDLE
from tensorized_layers.TP import TP


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
        ignore_modes=(0, 1, 2),
        Tensorized_mlp=True,
        tensor_type=("tle", "tle"),
        tdle_level=3,
    ):
        super(Encoder, self).__init__()
        self.tensorized_mlp = Tensorized_mlp
        self.ignore_modes = ignore_modes
        self.bias = bias
        self.device = device
        self.mlp_dim = mlp_dim
        self.embed_dim = embed_dim
        self.input_size = input_size

        self.tensor_input_size = (
            input_size[0],
            input_size[2] // patch_size + 1,
            input_size[3] // patch_size,
            embed_dim[0],
            embed_dim[1],
            embed_dim[2],
        )

        self.tensor_input_size_mid_layer = (
            input_size[0],
            input_size[2] // patch_size + 1,
            input_size[3] // patch_size,
            mlp_dim[0],
            mlp_dim[1],
            mlp_dim[2],
        )

        # self.trl_ranks = tuple([i for i in mlp_dim + embed_dim])

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
            tensor_type=tensor_type[0],
            tdle_level=tdle_level,
        )

        if self.tensorized_mlp == False:
            feature_dim = self.embed_dim[0] * self.embed_dim[1] * self.embed_dim[2]
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, self.mlp_dim),
                nn.GELU(),
                nn.Linear(self.mlp_dim, feature_dim),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(feature_dim)
        elif self.tensorized_mlp == True:
            if tensor_type[0] == "tle":
                tensor_layer_1 = TLE(
                    input_size=self.tensor_input_size,
                    rank=mlp_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                )
            elif tensor_type[0] == "tp":
                tensor_layer_1 = TP(
                    input_size=self.tensor_input_size,
                    output=self.mlp_dim,
                    rank=self.embed_dim + self.mlp_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                )
            else:
                tensor_layer_1 = TDLE(
                    input_size=self.tensor_input_size,
                    rank=mlp_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                    tdle_level=tdle_level,
                )

            if tensor_type[1] == "tle":
                tensor_layer_2 = TLE(
                    input_size=self.tensor_input_size_mid_layer,
                    rank=embed_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                )
            elif tensor_type[1] == "tp":
                tensor_layer_2 = TP(
                    input_size=self.tensor_input_size_mid_layer,
                    output=self.embed_dim,
                    rank=self.mlp_dim + self.embed_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                )
            else:
                tensor_layer_2 = TDLE(
                    input_size=self.tensor_input_size_mid_layer,
                    rank=embed_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                    tdle_level=tdle_level,
                )

            self.mlp = nn.Sequential(
                tensor_layer_1,
                nn.GELU(),
                tensor_layer_2,
                nn.Dropout(dropout),
            )
        else:
            self.mlp = self.tensorized_mlp[0]
            self.norm2 = self.tensorized_mlp[1]

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        if self.tensorized_mlp == False:
            shapes = x.shape
            x = rearrange(x, "b p1 p2 h w c -> b (p1 p2) (h w c)")
            x = x + self.dropout(self.mlp(self.norm2(x)))
            x = rearrange(
                x,
                "b (p1 p2) (h w c) -> b p1 p2 h w c",
                p1=shapes[1],
                p2=shapes[2],
                h=shapes[3],
                w=shapes[4],
                c=shapes[5],
            )
        else:
            x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
