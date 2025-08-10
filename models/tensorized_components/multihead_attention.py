import sys
from typing import Iterable, Optional, Sequence, Tuple, Union

sys.path.append(".")

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MultiHeadAttention(nn.Module):
    """
    Tensorized multi-head attention on factorized patch embeddings.

    Expects `x` of shape `(B, P_h + cls_tokens, P_w, d1, d2, d3)`. Q/K/V are produced by a
    tensorized layer (TLE/TDLE/TP), split into heads along each mode, attention is applied
    over the `P_h * P_w` spatial sequence, then optionally projected back via a tensorized layer.

    Parameters
    ----------
    input_size : Sequence[int]
        `(B, C, H, W)` of the original image; used to derive `(P_h, P_w)`.
    patch_size : int
        Patch size (H and W must be divisible by this).
    embed_dim : Tuple[int, int, int]
        Per-mode dims `(d1, d2, d3)` before reduction.
    num_heads : Tuple[int, int, int]
        Heads per mode `(h1, h2, h3)`. Each effective dim must be divisible by its heads.
    bias : bool, default True
        Whether TLE/TDLE/TP include bias.
    out_embed : bool, default True
        Apply an output tensorized projection after attention.
    ignore_modes : Iterable[int], default (0, 1, 2)
        Ignored modes for the tensorized layers (0=B, 1=P_h(+cls), 2=P_w).
    tensor_method : {'tle','tdle','tp'}, default 'tle'
        Tensor mapping used for Q/K/V/(out).
    tdle_level : int, default 3
        Number of summed TLE blocks inside TDLE.
    reduce_level : Tuple[int,int,int], default (0,0,0)
        Effective dims = embed_dim - reduce_level.
    cls_tokens : int, default 1
        Special rows prepended along P_h (e.g., CLS).
    attn_dropout : float, default 0.0
        Dropout on attention weights.
    proj_dropout : float, default 0.0
        Dropout after output projection.
    return_attn : bool, default False
        If True, returns `(y, attn)` where `attn` is (B, h1, h2, h3, seq, seq).
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

        def make_layer(method: str) -> nn.Module:
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
            rank = self.tensor_input_size[-3:] + self.embed_dim
            return TP(
                input_size=self.tensor_input_size,
                output_size=self.embed_dim,
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
        Parameters
        ----------
        x : torch.Tensor
            Input of shape `(B, P_h + cls_tokens, P_w, d1, d2, d3)`.
        attn_mask : torch.Tensor | None
            Broadcastable to `(B, h1, h2, h3, seq, seq)` with `seq = (P_h + cls_tokens) * P_w`.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Output `y` of shape `(B, P_h + cls_tokens, P_w, d1, d2, d3)`, and optionally attention.
        """
        if x.dim() != 6:
            raise ValueError(f"Expected 6D input (B, P_h+cls_tokens, P_w, d1, d2, d3), got {x.shape}")
        Bx, P1x, P2x, dx1, dx2, dx3 = x.shape
        B, P1, P2, d1, d2, d3 = self.tensor_input_size
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

        if self.return_attn:
            return y, attn
        return y


def _tensor_layer_flops(
    method: str,
    tensor_in: Tuple[int, int, int, int, int, int],
    embed_dim: Tuple[int, int, int],
    ignore_modes: Iterable[int],
    bias: bool,
    tdle_level: int,
) -> int:
    """
    Analytic FLOPs for a single tensor layer matching the configured method.
    """
    B, P1, P2, d1, d2, d3 = tensor_in
    B_eff = B * P1 * P2

    if method == "tp":
        rank = (d1 * 0 + 0,)  # placeholder; compute properly below
        rank = (  # input ranks = original image C,H,W; output ranks = embed_dim
            # In TP here we used rank as (C, H, W, d1, d2, d3) when creating the module.
            # We do not have C,H,W here; FLOPs estimator only needs input/output ranks:
            # use the module’s internal rank at call site (we’ll pass it via estimate_tp_flops).
        )
        raise RuntimeError("TP FLOPs for tensor attention are computed via estimate_tp_flops in the caller.")

    if method == "tle":
        proj = tle_input_projector_flops(tensor_in, embed_dim, ignore_modes)
        b = bias_add_flops(B_eff, embed_dim) if bias else 0
        return int(proj + b)

    if method == "tdle":
        proj = tle_input_projector_flops(tensor_in, embed_dim, ignore_modes)
        b = bias_add_flops(B_eff, embed_dim) if bias else 0
        per = int(proj + b)
        sum_cost = (tdle_level - 1) * B_eff * math.prod(embed_dim)
        return int(tdle_level * per + sum_cost)

    raise ValueError(f"Unknown tensor method: {method}")


def _sanity_check_once(
    input_size: Sequence[int],
    patch_size: int,
    embed_dim: Tuple[int, int, int],
    num_heads: Tuple[int, int, int],
    tensor_method: str = "tle",
    ignore_modes: Iterable[int] = (0, 1, 2),
    tdle_level: int = 2,
    reduce_level: Tuple[int, int, int] = (0, 0, 0),
    bias: bool = True,
    out_embed: bool = True,
    return_attn: bool = False,
    include_softmax: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run shape, parameter-count (total and tensorized-only), timing, and analytic FLOPs check.
    """
    device = device if device is not None else torch.device("cpu")

    attn = MultiHeadAttention(
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
        cls_tokens=1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        return_attn=return_attn,
    ).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(attn)

    inner_total = 0
    for m in (attn.tensor_layer_Q, attn.tensor_layer_K, attn.tensor_layer_V):
        t, _ = param_counts(m)
        inner_total += t
    if attn.tensor_layer_out is not None:
        t, _ = param_counts(attn.tensor_layer_out)
        inner_total += t

    print(
        f"[TensorMHA/{tensor_method}] Params: total={total_params:,}, "
        f"trainable={trainable_params:,} | tensorized-only={inner_total:,}"
    )

    B, C, H, W = map(int, input_size)
    P_h, P_w = H // patch_size, W // patch_size
    x = torch.randn(B, P_h + 1, P_w, *embed_dim, device=device, dtype=dtype, requires_grad=True)

    for _ in range(warmup):
        _ = attn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y = attn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    assert tuple(y.shape) == tuple(x.shape)

    y.mean().backward()
    assert x.grad is not None and tuple(x.grad.shape) == tuple(x.shape)

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[TensorMHA/{tensor_method}] Input shape:  {tuple(x.shape)}")
    print(f"[TensorMHA/{tensor_method}] Output shape: {tuple(y.shape)}")
    print(f"[TensorMHA/{tensor_method}] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    B_eff = B * (P_h + 1) * P_w
    h1, h2, h3 = num_heads
    xh, yh, zh = embed_dim[0] // h1, embed_dim[1] // h2, embed_dim[2] // h3
    seq = (P_h + 1) * P_w
    heads_total = h1 * h2 * h3
    d_per_head = xh * yh * zh

    if tensor_method == "tp":
        rank = embed_dim + embed_dim
        fl_q = estimate_tp_flops(
            input_size=(B, P_h + 1, P_w, *embed_dim),
            output_size=embed_dim,
            rank=rank,
            ignore_modes=ignore_modes,
            include_bias=bias,
        )["total"]
        fl_k = fl_q
        fl_v = fl_q
        fl_out = estimate_tp_flops(
            input_size=(B, P_h + 1, P_w, *embed_dim),
            output_size=embed_dim,
            rank=rank,
            ignore_modes=ignore_modes,
            include_bias=bias,
        )["total"] if out_embed else 0
    else:
        tensor_in = (B, P_h + 1, P_w, *embed_dim)
        fl_q = _tensor_layer_flops(tensor_method, tensor_in, embed_dim, ignore_modes, bias, tdle_level)
        fl_k = fl_q
        fl_v = fl_q
        fl_out = _tensor_layer_flops(tensor_method, tensor_in, embed_dim, ignore_modes, bias, tdle_level) if out_embed else 0

    fl_qk = 2 * B * heads_total * seq * seq * d_per_head
    fl_av = 2 * B * heads_total * seq * seq * d_per_head
    fl_softmax = (2 * B * heads_total * seq * seq) if include_softmax else 0

    total_flops = fl_q + fl_k + fl_v + fl_qk + fl_av + fl_softmax + fl_out

    print(f"[TensorMHA/{tensor_method}] FLOPs Q:         {to_gflops(fl_q):.3f} GFLOPs")
    print(f"[TensorMHA/{tensor_method}] FLOPs K:         {to_gflops(fl_k):.3f} GFLOPs")
    print(f"[TensorMHA/{tensor_method}] FLOPs V:         {to_gflops(fl_v):.3f} GFLOPs")
    print(f"[TensorMHA/{tensor_method}] FLOPs QK^T:      {to_gflops(fl_qk):.3f} GFLOPs")
    print(f"[TensorMHA/{tensor_method}] FLOPs A·V:       {to_gflops(fl_av):.3f} GFLOPs")
    if fl_softmax:
        print(f"[TensorMHA/{tensor_method}] FLOPs Softmax:   {to_gflops(fl_softmax):.3f} GFLOPs")
    if out_embed:
        print(f"[TensorMHA/{tensor_method}] FLOPs OutProj:   {to_gflops(fl_out):.3f} GFLOPs")
    print(f"[TensorMHA/{tensor_method}] FLOPs TOTAL:     {to_gflops(total_flops):.3f} GFLOPs")


def sanity_check() -> None:
    """
    Run tensorized MultiHeadAttention sanity checks for TLE, TDLE, and TP (bias on/off).
    """
    B, C, H, W = 256, 3, 32, 32
    ps = 8
    P_h, P_w = H // ps, W // ps
    embed_dim = (4, 4, 4)
    heads = (2, 2, 2)
    ignore = (0, 1, 2)
    tdle_level = 2

    for method in ("tle", "tdle", "tp"):
        _sanity_check_once((B, C, H, W), ps, embed_dim, heads, tensor_method=method, ignore_modes=ignore, tdle_level=tdle_level, bias=True, out_embed=True)
        _sanity_check_once((B, C, H, W), ps, embed_dim, heads, tensor_method=method, ignore_modes=ignore, tdle_level=tdle_level, bias=False, out_embed=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
