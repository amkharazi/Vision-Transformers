import sys
from typing import Iterable, Optional, Sequence, Tuple

sys.path.append(".")

import time
import math
import torch
import torch.nn as nn
from einops import rearrange

from tensorized_layers.TLE import TLE
from tensorized_layers.TP import TP
from tensorized_layers.TDLE import TDLE

from utils.num_param import param_counts
from utils.flops import (
    estimate_tp_flops,
    tle_input_projector_flops,
    bias_add_flops,
    to_gflops,
)


class PatchEmbedding(nn.Module):
    """
    Tensorized patch embedding layer.

    Splits an image into non-overlapping patches of size (patch_size x patch_size),
    reshapes to (B, P_h, P_w, C, p, p), and applies a tensorized mapping using
    TLE, TDLE, or TP.

    Parameters
    ----------
    input_size : Sequence[int]
        Shape (B, C, H, W) of the input.
    patch_size : int
        Size of each square patch. H and W must be divisible by this value.
    embed_dim : Tuple[int, int, int]
        Output embedding mode sizes (d1, d2, d3). Used as the rank for TLE/TDLE and as the output size for TP.
    bias : bool, default True
        Whether to include bias in the tensorized layer.
    ignore_modes : Iterable[int], default (0, 1, 2)
        Modes to ignore in the tensor decomposition.
    tensor_method : {'tle', 'tdle', 'tp'}, default 'tle'
        Which tensorized layer to use.
    tdle_level : int, default 3
        Hierarchy level for TDLE.
    rank : Optional[Sequence[int]], default None
        Rank vector used only when tensor_method == 'tp'. If None, defaults to concatenating
        input modes and output modes as rank = (*in_modes, *out_modes). Ignored for 'tle' and 'tdle'.
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
        rank: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        if not isinstance(embed_dim, tuple) or len(embed_dim) != 3:
            raise TypeError(f"embed_dim must be a 3-tuple, got {embed_dim}")
        if tensor_method not in {"tle", "tdle", "tp"}:
            raise ValueError(f"Invalid tensor_method '{tensor_method}'")
        if not (isinstance(input_size, Sequence) and len(input_size) == 4):
            raise TypeError(f"input_size must be (B, C, H, W), got {input_size}")

        B, C, H, W = map(int, input_size)
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"H and W must be divisible by patch_size={patch_size}")
        if any(int(d) <= 0 for d in embed_dim):
            raise ValueError(f"embed_dim entries must be > 0, got {embed_dim}")

        self.input_size = (B, C, H, W)
        self.patch_size = patch_size
        self.embed_dim = tuple(int(d) for d in embed_dim)
        self.bias = bias
        self.ignore_modes = tuple(ignore_modes)
        self.tensor_method = tensor_method

        P_h, P_w = H // patch_size, W // patch_size
        self.tensor_input_size = (B, P_h, P_w, C, patch_size, patch_size)

        def resolve_tp_rank(r: Optional[Sequence[int]]) -> Tuple[int, ...]:
            if r is not None:
                return tuple(int(x) for x in r)
            in_modes = self.tensor_input_size[3:]
            out_modes = self.embed_dim
            return tuple(int(m) for m in (*in_modes, *out_modes))

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
            tp_rank = resolve_tp_rank(rank)
            expected_len = len(self.tensor_input_size[3:]) + len(self.embed_dim)
            if len(tp_rank) != expected_len:
                raise ValueError(f"rank length must be {expected_len} for TP, got {len(tp_rank)}")
            if any(int(r) <= 0 for r in tp_rank):
                raise ValueError(f"rank entries must be > 0, got {tp_rank}")
            self.tensor_layer = TP(
                input_size=self.tensor_input_size,
                output_size=self.embed_dim,
                rank=tp_rank,
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output of shape (B, P_h, P_w, d1, d2, d3) where (d1, d2, d3) == embed_dim.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W), got {x.shape}")
        Bx, Cx, Hx, Wx = x.shape
        B, C, H, W = self.input_size
        if (Cx, Hx, Wx) != (C, H, W):
            raise ValueError(f"Expected (C,H,W)=({C},{H},{W}), got ({Cx},{Hx},{Wx})")

        x = rearrange(
            x,
            "b c (p1 h) (p2 w) -> b p1 p2 c h w",
            h=self.patch_size,
            w=self.patch_size,
        )
        return self.tensor_layer(x)


def _estimate_tensor_flops(
    tensor_method: str,
    input_size: Sequence[int],
    patch_size: int,
    embed_dim: Tuple[int, int, int],
    ignore_modes: Iterable[int],
    bias: bool,
    tdle_level: int,
    rank: Optional[Sequence[int]] = None,
) -> Tuple[int, str]:
    """
    Analytic FLOPs for the configured tensor layer; returns (flops, detail_tag).
    """
    B, C, H, W = map(int, input_size)
    P_h, P_w = H // patch_size, W // patch_size
    tensor_in = (B, P_h, P_w, C, patch_size, patch_size)
    B_eff = B * P_h * P_w

    if tensor_method == "tp":
        if rank is None:
            in_modes = tensor_in[3:]
            out_modes = embed_dim
            rank = tuple(int(m) for m in (*in_modes, *out_modes))
        out_size = tuple(int(x) for x in embed_dim)
        parts = estimate_tp_flops(
            input_size=tensor_in,
            output_size=out_size,
            rank=tuple(int(r) for r in rank),
            ignore_modes=ignore_modes,
            include_bias=bias,
        )
        return int(parts["total"]), "TP(total)"

    if tensor_method == "tle":
        proj = tle_input_projector_flops(tensor_in, embed_dim, ignore_modes)
        b = bias_add_flops(B_eff, embed_dim) if bias else 0
        return int(proj + b), "TLE(projector+bias)"

    if tensor_method == "tdle":
        proj = tle_input_projector_flops(tensor_in, embed_dim, ignore_modes)
        b = bias_add_flops(B_eff, embed_dim) if bias else 0
        per = int(proj + b)
        sum_cost = (tdle_level - 1) * B_eff * math.prod(embed_dim)
        return int(tdle_level * per + sum_cost), "TDLE(r*per + sums)"

    raise ValueError(f"Unknown tensor_method {tensor_method}")


def _sanity_check_once(
    input_size: Sequence[int],
    patch_size: int,
    embed_dim: Tuple[int, int, int],
    tensor_method: str = "tle",
    ignore_modes: Iterable[int] = (0, 1, 2),
    tdle_level: int = 3,
    rank: Optional[Sequence[int]] = None,
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run shape, parameter-count (total and tensorized-only), timing, and analytic FLOPs check.
    """
    device = device if device is not None else torch.device("cpu")

    pe = PatchEmbedding(
        input_size=input_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        bias=bias,
        ignore_modes=ignore_modes,
        tensor_method=tensor_method,
        tdle_level=tdle_level,
        rank=rank,
    ).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(pe)
    inner_total, inner_trainable = param_counts(pe.tensor_layer)
    print(
        f"[PatchEmbedding/{tensor_method}] Params: total={total_params:,}, "
        f"trainable={trainable_params:,} | tensorized-only={inner_total:,}"
    )

    B, C, H, W = map(int, input_size)
    x = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=True)

    for _ in range(warmup):
        _ = pe(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y = pe(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    P_h, P_w = H // patch_size, W // patch_size
    out_rank = tuple(int(a) for a in embed_dim)
    assert tuple(y.shape) == (B, P_h, P_w, *out_rank)

    y.mean().backward()
    assert x.grad is not None and tuple(x.grad.shape) == (B, C, H, W)

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[PatchEmbedding/{tensor_method}] Input shape:  {(B, C, H, W)}")
    print(f"[PatchEmbedding/{tensor_method}] Output shape: {(B, P_h, P_w, *out_rank)}")
    print(f"[PatchEmbedding/{tensor_method}] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    flops, detail = _estimate_tensor_flops(
        tensor_method=tensor_method,
        input_size=input_size,
        patch_size=patch_size,
        embed_dim=out_rank,
        ignore_modes=ignore_modes,
        bias=bias,
        tdle_level=tdle_level,
        rank=rank,
    )
    print(f"[PatchEmbedding/{tensor_method}] FLOPs {detail}: {to_gflops(flops):.3f} GFLOPs")


def sanity_check() -> None:
    """
    Run PatchEmbedding sanity checks for TLE, TDLE, and TP (bias on/off).
    """
    B, C, H, W = 256, 3, 32, 32
    ps = 4
    embed_dim = (4, 4, 4)
    ignore = (0, 1, 2)
    tdle_level = 3

    methods = ("tle", "tdle", "tp")
    for method in methods:
        tp_rank = None
        _sanity_check_once((B, C, H, W), ps, embed_dim, tensor_method=method, ignore_modes=ignore, tdle_level=tdle_level, rank=tp_rank, bias=True)
        _sanity_check_once((B, C, H, W), ps, embed_dim, tensor_method=method, ignore_modes=ignore, tdle_level=tdle_level, rank=tp_rank, bias=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
