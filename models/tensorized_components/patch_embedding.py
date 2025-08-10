import sys
from typing import Iterable, Sequence, Tuple

sys.path.append(".")

import torch
import torch.nn as nn
from einops import rearrange

from tensorized_layers.TLE import TLE
from tensorized_layers.TP import TP
from tensorized_layers.TDLE import TDLE


class PatchEmbedding(nn.Module):
    """
    Tensorized patch embedding layer.

    Splits an image into non-overlapping patches of size (patch_size x patch_size),
    reshapes to (B, P_h, P_w, C, p, p), and applies a tensorized mapping using
    TLE, TDLE, or TP.

    Parameters
    ----------
    input_size : Sequence[int]
        (B, C, H, W) of the input.
    patch_size : int
        Size of each patch (H and W must be divisible by this).
    embed_dim : Tuple[int, int, int]
        Factorized embedding rank (d1, d2, d3), adjusted internally by reduce_level.
    bias : bool, default True
        Whether to include bias in the tensorized layer.
    ignore_modes : Iterable[int], default (0, 1, 2)
        Modes to ignore in the tensor decomposition.
    tensor_method : {'tle', 'tdle', 'tp'}, default 'tle'
        Which tensorized layer to use.
    tdle_level : int, default 3
        Hierarchy level for TDLE.
    reduce_level : Tuple[int, int, int], default (0, 0, 0)
        Per-mode reduction applied to embed_dim.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        patch_size: int,
        embed_dim: Tuple[int, int, int],
        bias: bool = True,
        ignore_modes: Iterable[int] = (0, 1, 2),
        tensor_method: str = "tle",
        tdle_level: int = 3,
        reduce_level: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()

        if not isinstance(embed_dim, tuple) or len(embed_dim) != 3:
            raise TypeError(f"embed_dim must be a 3-tuple, got {embed_dim}")
        if not isinstance(reduce_level, tuple) or len(reduce_level) != 3:
            raise TypeError(f"reduce_level must be a 3-tuple, got {reduce_level}")
        if tensor_method not in {"tle", "tdle", "tp"}:
            raise ValueError(f"Invalid tensor_method '{tensor_method}'")
        if not (isinstance(input_size, Sequence) and len(input_size) == 4):
            raise TypeError(f"input_size must be (B, C, H, W), got {input_size}")

        B, C, H, W = map(int, input_size)
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"H and W must be divisible by patch_size={patch_size}")

        _embed_dim = tuple(x - y for x, y in zip(embed_dim, reduce_level))
        if any(d <= 0 for d in _embed_dim):
            raise ValueError(f"embed_dim - reduce_level must be > 0, got {_embed_dim}")

        self.input_size = (B, C, H, W)
        self.patch_size = patch_size
        self.embed_dim = _embed_dim
        self.bias = bias
        self.ignore_modes = tuple(ignore_modes)
        self.tensor_method = tensor_method

        P_h, P_w = H // patch_size, W // patch_size
        self.tensor_input_size = (B, P_h, P_w, C, patch_size, patch_size)

        if tensor_method == "tdle":
            self.tensor_layer = TDLE(
                input_size=self.tensor_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
                r=tdle_level,
            )
        elif tensor_method == "tle":
            self.tensor_layer = TLE(
                input_size=self.tensor_input_size,
                rank=self.embed_dim,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
            )
        elif tensor_method == "tp":
            rank = self.input_size[-3:] + self.embed_dim
            output_size = tuple(x + y for x, y in zip(self.embed_dim, reduce_level))
            self.tensor_layer = TP(
                input_size=self.tensor_input_size,
                output_size=output_size,
                rank=rank,
                ignore_modes=self.ignore_modes,
                bias=self.bias,
            )

    @property
    def num_patches(self) -> int:
        _, _, H, W = self.input_size
        return (H // self.patch_size) * (W // self.patch_size)

    @property
    def patch_grid(self) -> Tuple[int, int]:
        _, _, H, W = self.input_size
        return (H // self.patch_size, W // self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W), got {x.shape}")
        Bx, Cx, Hx, Wx = x.shape
        B, C, H, W = self.input_size
        if (Cx, Hx, Wx) != (C, H, W):
            raise ValueError(
                f"Expected (C,H,W)=({C},{H},{W}), got ({Cx},{Hx},{Wx})"
            )

        x = rearrange(
            x,
            "b c (p1 h) (p2 w) -> b p1 p2 c h w",
            h=self.patch_size,
            w=self.patch_size,
        )
        return self.tensor_layer(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    B, C, H, W = 256, 3, 32, 32
    ps = 4
    embed_dim = (4, 4, 4)
    reduce_level = (0, 0, 0)
    P_h, P_w = H // ps, W // ps

    x = torch.randn(B, C, H, W, requires_grad=True)

    pe_tle = PatchEmbedding((B, C, H, W), ps, embed_dim, tensor_method="tle")
    out_tle = pe_tle(x)
    assert out_tle.shape == (B, P_h, P_w, *embed_dim)
    out_tle.mean().backward(retain_graph=True)
    assert x.grad is not None

    x.grad = None
    pe_tdle = PatchEmbedding((B, C, H, W), ps, embed_dim, tensor_method="tdle")
    out_tdle = pe_tdle(x.detach().requires_grad_(True))
    assert out_tdle.shape == (B, P_h, P_w, *embed_dim)
    out_tdle.sum().backward()

    x_tp = torch.randn(B, C, H, W, requires_grad=True)
    pe_tp = PatchEmbedding((B, C, H, W), ps, embed_dim, tensor_method="tp")
    out_tp = pe_tp(x_tp)
    out_tp.mean().backward()
    assert x_tp.grad is not None

    print("All sanity checks passed.")
