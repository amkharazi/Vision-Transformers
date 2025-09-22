import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensorized_layers.TLE import TLE
from tensorized_layers.TDLE import TDLE
from tensorized_layers.TP import TP


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_size,
        patch_size,
        embed_dim,
        num_heads,
        bias=True,
        out_embed=True,
        device="cuda",
        ignore_modes=(0, 1, 2),
        tensor_type="tle",
        tdle_level=3,
    ):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        self.bias = bias
        self.ignore_modes = ignore_modes

        self.h1 = num_heads[0]
        self.h2 = num_heads[1]
        self.h3 = num_heads[2]

        # TO DO : requires improvement
        self.scale = (
            (self.embed_dim[0] // self.h1)
            * (self.embed_dim[1] // self.h2)
            * (self.embed_dim[2] // self.h3)
        ) ** -0.5

        self.out_embed = out_embed

        self.tcl_input_size = (
            self.input_size[0],
            self.input_size[2] // self.patch_size + 1,
            self.input_size[3] // self.patch_size,
            self.embed_dim[0],
            self.embed_dim[1],
            self.embed_dim[2],
        )

        if tensor_type == "tle":
            self.tensor_layer_q = TLE(
                input_size=self.tcl_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
            )
            self.tensor_layer_k = TLE(
                input_size=self.tcl_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
            )
            self.tensor_layer_v = TLE(
                input_size=self.tcl_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
            )
            if self.out_embed:
                self.tensor_layer_out = TLE(
                    input_size=self.tcl_input_size,
                    rank=self.embed_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                )
        elif tensor_type == "TP":
            self.tensor_layer_q = TP(
                input_size=self.tcl_input_size,
                output=self.embed_dim,
                rank=self.embed_dim * 2,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
            )
            self.tensor_layer_k = TP(
                input_size=self.tcl_input_size,
                output=self.embed_dim,
                rank=self.embed_dim * 2,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
            )
            self.tensor_layer_v = TP(
                input_size=self.tcl_input_size,
                output=self.embed_dim,
                rank=self.embed_dim * 2,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
            )
            if self.out_embed:
                self.tensor_layer_out = TP(
                    input_size=self.tcl_input_size,
                    output=self.embed_dim,
                    rank=self.embed_dim * 2,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                )
        else:
            self.tensor_layer_q = TDLE(
                input_size=self.tcl_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
                r=tdle_level,
            )
            self.tensor_layer_k = TDLE(
                input_size=self.tcl_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
                r=tdle_level,
            )
            self.tensor_layer_v = TDLE(
                input_size=self.tcl_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                device=self.device,
                r=tdle_level,
            )
            if self.out_embed:
                self.tensor_layer_out = TDLE(
                    input_size=self.tcl_input_size,
                    rank=self.embed_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    device=self.device,
                    r=tdle_level,
                )

    def forward(self, x):
        q = self.tensor_layer_q(x)
        k = self.tensor_layer_k(x)
        v = self.tensor_layer_v(x)

        Q = rearrange(
            q,
            "b p1 p2 (x h1) (y h2) (z h3) -> b p1 p2 h1 h2 h3 x y z",
            h1=self.h1,
            h2=self.h2,
            h3=self.h3,
        )
        K = rearrange(
            k,
            "b p1 p2 (x h1) (y h2) (z h3) -> b p1 p2 h1 h2 h3 x y z",
            h1=self.h1,
            h2=self.h2,
            h3=self.h3,
        )
        V = rearrange(
            v,
            "b p1 p2 (x h1) (y h2) (z h3) -> b p1 p2 h1 h2 h3 x y z",
            h1=self.h1,
            h2=self.h2,
            h3=self.h3,
        )

        attn = (
            torch.einsum("b p q d e f x y z,b m n d e f x y z -> b d e f p q m n", Q, K)
            * self.scale
        )

        softmax_attn = rearrange(
            attn, "b h1 h2 h3 p1 p2 q1 q2 -> b (h1 h2 h3) (p1 p2) (q1 q2)"
        )
        softmax_attn = F.softmax(softmax_attn, dim=-1)
        attention = softmax_attn.view_as(attn)

        x = torch.einsum(
            "b d e f p q m n , b m n d e f x y z -> b d e f p q x y z ", attention, V
        )
        x = rearrange(
            x,
            "b h1 h2 h3 p1 p2 x y z  -> b p1 p2 (x h1) (y h2) (z h3)",
            h1=self.h1,
            h2=self.h2,
            h3=self.h3,
        )

        if self.out_embed:
            x = self.tensor_layer_out(x)

        return x
