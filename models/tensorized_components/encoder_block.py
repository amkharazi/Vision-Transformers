import sys
from typing import Iterable, Sequence, Tuple
from typing import Optional

sys.path.append(".")

import torch
import torch.nn as nn
from einops import rearrange  # noqa: F401

from models.tensorized_components.multihead_attention import MultiHeadAttention as MHA
from tensorized_layers.TLE import TLE
from tensorized_layers.TDLE import TDLE
from tensorized_layers.TP import TP


class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob or 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Encoder(nn.Module):
    """
    Transformer encoder block for tensorized tokens.

    Expects input x of shape (B, P_h+1, P_w, d1, d2, d3). Applies LayerNorm → MHA (+residual with DropPath),
    then LayerNorm → MLP (+residual with DropPath). MLP is two tensorized linear layers with GELU+Dropout.

    Parameters
    ----------
    input_size : Sequence[int]
        (B, C, H, W) of the original image; used to derive patch grid size for tensorized layers.
    patch_size : int
        Patch size (H and W must be divisible by this).
    embed_dim : Tuple[int, int, int]
        Token embedding dims (d1, d2, d3) before reduction.
    num_heads : Tuple[int, int, int]
        Heads per mode for MHA.
    mlp_dim : Tuple[int, int, int]
        Hidden dims for the MLP (before reduction).
    dropout : float, default 0.5
    bias : bool, default True
    out_embed : bool, default True
        Whether MHA has an output projection.
    drop_path : float, default 0.1
    ignore_modes : Iterable[int], default (0,1,2)
    tensor_method_mlp : Tuple[str, str], default ('tle','tle')
        Methods for the two MLP tensorized layers.
    tensor_method : str, default 'tle'
        Method for MHA's Q/K/V/out projections.
    tdle_level : int, default 3
    reduce_level : Tuple[int,int,int], default (0,0,0)
        Effective dims = dims - reduce_level.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        embed_dim: Tuple[int, int, int],
        num_heads: Tuple[int, int, int],
        mlp_dim: Tuple[int, int, int],
        dropout: float = 0.5,
        bias: bool = True,
        out_embed: bool = True,
        drop_path: float = 0.1,
        ignore_modes: Iterable[int] = (0, 1, 2),
        tensor_method_mlp: Tuple[str, str] = ("tle", "tle"),
        tensor_method: str = "tle",
        tdle_level: int = 3,
        reduce_level: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()

        if not (isinstance(tensor_method_mlp, tuple) and len(tensor_method_mlp) == 2):
            raise TypeError("tensor_method_mlp must be a tuple of length 2")
        if any(m not in {"tle", "tdle", "tp"} for m in tensor_method_mlp):
            raise ValueError(f"Invalid methods in tensor_method_mlp: {tensor_method_mlp}")
        if not (isinstance(input_size, Sequence) and len(input_size) == 4):
            raise TypeError(f"input_size must be (B, C, H, W), got {input_size}")

        B, C, H, W = map(int, input_size)
        if H % patch_size or W % patch_size:
            raise ValueError(f"H and W must be divisible by patch_size={patch_size}")

        eff_embed = tuple(e - r for e, r in zip(embed_dim, reduce_level))
        eff_mlp = tuple(m - r for m, r in zip(mlp_dim, reduce_level))
        if any(d <= 0 for d in eff_embed + eff_mlp):
            raise ValueError(f"embed_dim/mlp_dim - reduce_level must be > 0, got {eff_embed}, {eff_mlp}")

        self.input_size = (B, C, H, W)
        self.patch_size = int(patch_size)
        self.embed_dim = eff_embed
        self.mlp_dim = eff_mlp
        self.bias = bool(bias)
        self.ignore_modes = tuple(ignore_modes)
        self.reduce_level = reduce_level

        P_h, P_w = H // self.patch_size, W // self.patch_size
        self.tensor_input_size_layer1 = (B, P_h + 1, P_w, *self.embed_dim)
        self.tensor_input_size_layer2 = (B, P_h + 1, P_w, *self.mlp_dim)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.attention = MHA(
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            out_embed=out_embed,
            ignore_modes=ignore_modes,
            tensor_method=tensor_method,
            tdle_level=tdle_level,
            reduce_level=reduce_level,
        )

        if tensor_method_mlp[0] == "tdle":
            layer1 = TDLE(self.tensor_input_size_layer1, self.mlp_dim, ignore_modes, bias, r=tdle_level)
        elif tensor_method_mlp[0] == "tle":
            layer1 = TLE(self.tensor_input_size_layer1, self.mlp_dim, ignore_modes, bias)
        else:  # 'tp'
            rank = self.input_size[-3:] + self.mlp_dim
            output_size = tuple(x + y for x, y in zip(self.mlp_dim, reduce_level))
            layer1 = TP(
                input_size=self.tensor_input_size_layer1,
                output_size=output_size,
                rank=rank,
                ignore_modes=ignore_modes,
                bias=bias,
            )

        if tensor_method_mlp[1] == "tdle":
            layer2 = TDLE(self.tensor_input_size_layer2, self.embed_dim, ignore_modes, bias, r=tdle_level)
        elif tensor_method_mlp[1] == "tle":
            layer2 = TLE(self.tensor_input_size_layer2, self.embed_dim, ignore_modes, bias)
        else:  # 'tp'
            rank = self.input_size[-3:] + self.embed_dim
            output_size = tuple(x + y for x, y in zip(self.embed_dim, reduce_level))
            layer2 = TP(
                input_size=self.tensor_input_size_layer2,
                output_size=output_size,
                rank=rank,
                ignore_modes=ignore_modes,
                bias=bias,
            )

        self.mlp = nn.Sequential(
            layer1,
            nn.GELU(),
            nn.Dropout(dropout),
            layer2,
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 256, 3, 32, 32
    ps = 8
    P_h, P_w = H // ps, W // ps
    embed_dim = (4, 4, 4)
    mlp_dim = (4, 4, 8)
    heads = (2, 2, 2)

    x = torch.randn(B, P_h + 1, P_w, *embed_dim, requires_grad=True)

    enc_tle = Encoder(
        input_size=(B, C, H, W),
        patch_size=ps,
        embed_dim=embed_dim,
        num_heads=heads,
        mlp_dim=mlp_dim,
        tensor_method="tle",
        tensor_method_mlp=("tle", "tle"),
    )
    y = enc_tle(x)
    assert y.shape == x.shape
    y.mean().backward(retain_graph=True)
    assert x.grad is not None

    x.grad = None
    enc_tdle = Encoder(
        input_size=(B, C, H, W),
        patch_size=ps,
        embed_dim=embed_dim,
        num_heads=heads,
        mlp_dim=mlp_dim,
        tensor_method="tdle",
        tensor_method_mlp=("tdle", "tdle"),
        tdle_level=2,
    )
    y2 = enc_tdle(x.detach().requires_grad_(True))
    assert y2.shape == x.shape
    y2.sum().backward()

    x.grad = None
    enc_tp = Encoder(
        input_size=(B, C, H, W),
        patch_size=ps,
        embed_dim=embed_dim,
        num_heads=heads,
        mlp_dim=mlp_dim,
        tensor_method="tp",
        tensor_method_mlp=("tp", "tp"),
    )
    y3 = enc_tp(x.detach().requires_grad_(True))
    assert y3.shape == x.shape
    y3.sum().backward()

    print("Encoder sanity checks passed.")
