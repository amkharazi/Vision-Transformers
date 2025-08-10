import sys
from typing import Iterable, Sequence, Tuple
from typing import Union

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tensorized_layers.TLE import TLE
from tensorized_layers.TP import TP
from tensorized_layers.TDLE import TDLE


class MultiHeadAttention(nn.Module):
    """
    Tensorized multi-head attention on factorized patch embeddings.

    Expects x of shape (B, P_h+cls_tokens, P_w, d1, d2, d3). Q/K/V are produced by a
    tensorized layer (TLE/TDLE/TP), split into heads along each mode, attention is applied,
    then optionally projected back via another tensorized layer.

    Parameters
    ----------
    input_size : Sequence[int]
        (B, C, H, W) of the original image; used to derive (P_h, P_w).
    patch_size : int
        Patch size (H and W must be divisible by this).
    embed_dim : Tuple[int, int, int]
        Per-mode dims (d1, d2, d3) before reduction.
    num_heads : Tuple[int, int, int]
        Heads per mode (h1, h2, h3). Each effective dim must be divisible by corresponding heads.
    bias : bool, default True
    out_embed : bool, default True
        Apply an output tensorized projection after attention.
    ignore_modes : Iterable[int], default (0,1,2)
    tensor_method : {'tle','tdle','tp'}, default 'tle'
    tdle_level : int, default 3
    reduce_level : Tuple[int,int,int], default (0,0,0)
        Effective dims = embed_dim - reduce_level.
    cls_tokens : int, default 1
        Number of special rows prepended on P_h (e.g., CLS).
    attn_dropout : float, default 0.0
    proj_dropout : float, default 0.0
    return_attn : bool, default False
        If True, returns (y, attn).
    """

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        embed_dim: Tuple[int, int, int],
        num_heads: Tuple[int, int, int],
        bias: bool = True,
        out_embed: bool = True,
        ignore_modes: Iterable[int] = (0, 1, 2),
        tensor_method: str = "tle",
        tdle_level: int = 3,
        reduce_level: Tuple[int, int, int] = (0, 0, 0),
        cls_tokens: int = 1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        return_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (isinstance(embed_dim, tuple) and len(embed_dim) == 3):
            raise TypeError(f"embed_dim must be a 3-tuple, got {embed_dim}")
        if not (isinstance(reduce_level, tuple) and len(reduce_level) == 3):
            raise TypeError(f"reduce_level must be a 3-tuple, got {reduce_level}")
        if tensor_method not in {"tle", "tdle", "tp"}:
            raise ValueError(f"Invalid tensor_method '{tensor_method}'")
        if not (isinstance(input_size, Sequence) and len(input_size) == 4):
            raise TypeError(f"input_size must be (B, C, H, W), got {input_size}")
        if not (isinstance(num_heads, tuple) and len(num_heads) == 3):
            raise TypeError(f"num_heads must be a 3-tuple, got {num_heads}")
        if cls_tokens < 0:
            raise ValueError("cls_tokens must be >= 0")

        B, C, H, W = map(int, input_size)
        if H % patch_size or W % patch_size:
            raise ValueError(f"H and W must be divisible by patch_size={patch_size}")

        eff_dim = tuple(e - r for e, r in zip(embed_dim, reduce_level))
        if any(d <= 0 for d in eff_dim):
            raise ValueError(f"embed_dim - reduce_level must be > 0, got {eff_dim}")

        h1, h2, h3 = num_heads
        if eff_dim[0] % h1 or eff_dim[1] % h2 or eff_dim[2] % h3:
            raise ValueError(f"Each dim must be divisible by heads: eff_dim={eff_dim}, heads={num_heads}")

        self.input_size = (B, C, H, W)
        self.patch_size = int(patch_size)
        self.embed_dim = eff_dim
        self.num_heads = (h1, h2, h3)
        self.bias = bool(bias)
        self.out_embed = bool(out_embed)
        self.ignore_modes = tuple(ignore_modes)
        self.tensor_method = tensor_method
        self.cls_tokens = int(cls_tokens)
        self.return_attn = bool(return_attn)

        P_h, P_w = H // self.patch_size, W // self.patch_size
        self.tensor_input_size = (B, P_h + self.cls_tokens, P_w, *self.embed_dim)

        self.scale = (
            (self.embed_dim[0] // h1)
            * (self.embed_dim[1] // h2)
            * (self.embed_dim[2] // h3)
        ) ** -0.5

        def make_layer(method: str):
            if method == "tdle":
                return TDLE(
                    input_size=self.tensor_input_size,
                    rank=self.embed_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                    r=tdle_level,
                )
            if method == "tle":
                return TLE(
                    input_size=self.tensor_input_size,
                    rank=self.embed_dim,
                    ignore_modes=self.ignore_modes,
                    bias=self.bias,
                )
            rank = self.input_size[-3:] + self.embed_dim
            output_size = self.embed_dim
            return TP(
                input_size=self.tensor_input_size,
                output_size=output_size,
                rank=rank,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
            )

        self.tensor_layer_Q = make_layer(tensor_method)
        self.tensor_layer_K = make_layer(tensor_method)
        self.tensor_layer_V = make_layer(tensor_method)
        self.tensor_layer_out = make_layer(tensor_method) if self.out_embed else None

        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Union[torch.Tensor, None] = None
    ):
        """
        x : (B, P_h+cls_tokens, P_w, d1, d2, d3)
        attn_mask : optional, broadcastable to (B, h1, h2, h3, seq, seq)
        """
        if x.dim() != 6:
            raise ValueError(f"Expected 6D input (B, P_h+cls_tokens, P_w, d1, d2, d3), got {x.shape}")
        Bx, P1x, P2x, dx1, dx2, dx3 = x.shape
        B, P1, P2, d1, d2, d3 = self.tensor_input_size
        # if (Bx, P1x, P2x, dx1, dx2, dx3) != (B, P1, P2, d1, d2, d3):
        #     raise ValueError(f"Input shape mismatch: expected {self.tensor_input_size}, got {tuple(x.shape)}")
        
        if (P1x, P2x, dx1, dx2, dx3) != (P1, P2, d1, d2, d3):
            raise ValueError(f"Input shape mismatch: expected {self.tensor_input_size}, got {tuple(x.shape)} excluding the first batch mode!")

        q = self.tensor_layer_Q(x)
        k = self.tensor_layer_K(x)
        v = self.tensor_layer_V(x)

        h1, h2, h3 = self.num_heads
        Q = rearrange(q, "b p1 p2 (x h1) (y h2) (z h3) -> b h1 h2 h3 p1 p2 x y z", h1=h1, h2=h2, h3=h3)
        K = rearrange(k, "b p1 p2 (x h1) (y h2) (z h3) -> b h1 h2 h3 p1 p2 x y z", h1=h1, h2=h2, h3=h3)
        V = rearrange(v, "b p1 p2 (x h1) (y h2) (z h3) -> b h1 h2 h3 p1 p2 x y z", h1=h1, h2=h2, h3=h3)

        q2 = rearrange(Q, "b h1 h2 h3 p1 p2 x y z -> b h1 h2 h3 (p1 p2) (x y z)")
        k2 = rearrange(K, "b h1 h2 h3 p1 p2 x y z -> b h1 h2 h3 (p1 p2) (x y z)")
        v2 = rearrange(V, "b h1 h2 h3 p1 p2 x y z -> b h1 h2 h3 (p1 p2) (x y z)")

        attn = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        if attn_mask is not None:
            while attn_mask.dim() < attn.dim():
                attn_mask = attn_mask.unsqueeze(0)
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v2)
        y = rearrange(
            y,
            "b h1 h2 h3 (p1 p2) (x y z) -> b p1 p2 (x h1) (y h2) (z h3)",
            h1=h1, h2=h2, h3=h3, p1=V.shape[4], p2=V.shape[5], x=V.shape[6], y=V.shape[7], z=V.shape[8],
        )

        if self.tensor_layer_out is not None:
            y = self.tensor_layer_out(y)
        y = self.proj_drop(y)
        return y


if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 256, 3, 32, 32
    ps = 8
    P_h, P_w = H // ps, W // ps

    embed_dim = (4, 4, 4)
    heads = (2, 2, 2)

    x = torch.randn(B, P_h + 1, P_w, *embed_dim, requires_grad=True)

    attn_tle = MultiHeadAttention(
        input_size=(B, C, H, W),
        patch_size=ps,
        embed_dim=embed_dim,
        num_heads=heads,
        tensor_method="tle",
    )
    y_tle = attn_tle(x)
    assert y_tle.shape == x.shape
    y_tle.mean().backward(retain_graph=True)
    assert x.grad is not None

    x.grad = None
    attn_tdle = MultiHeadAttention(
        input_size=(B, C, H, W),
        patch_size=ps,
        embed_dim=embed_dim,
        num_heads=heads,
        tensor_method="tdle",
        tdle_level=2,
    )
    y_tdle = attn_tdle(x.detach().requires_grad_(True))
    assert y_tdle.shape == x.shape
    y_tdle.sum().backward()

    x.grad = None
    attn_tp = MultiHeadAttention(
        input_size=(B, C, H, W),
        patch_size=ps,
        embed_dim=embed_dim,
        num_heads=heads,
        tensor_method="tp",  # TP rank set internally as self.input_size[-3:] + self.embed_dim
    )
    y_tp = attn_tp(x.detach().requires_grad_(True))
    assert y_tp.shape == x.shape
    y_tp.sum().backward()

    print("MultiHeadAttention sanity checks passed.")
