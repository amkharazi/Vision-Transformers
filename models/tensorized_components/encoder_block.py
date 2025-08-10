import sys
from typing import Iterable, Optional, Sequence, Tuple

sys.path.append(".")

import time
import math
import torch
import torch.nn as nn
from einops import rearrange  # noqa: F401

from models.tensorized_components.multihead_attention import MultiHeadAttention as MHA
from tensorized_layers.TLE import TLE
from tensorized_layers.TDLE import TDLE
from tensorized_layers.TP import TP

from utils.num_param import param_counts
from utils.flops import (
    estimate_tp_flops,
    tle_input_projector_flops,
    bias_add_flops,
    layernorm_flops,
    residual_add_flops,
    droppath_flops,
    to_gflops,
)


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
    then LayerNorm → MLP (+residual with DropPath). MLP is two tensorized layers with GELU+Dropout.

    Parameters
    ----------
    input_size : Sequence[int]
        (B, C, H, W) of the original image; used to derive patch grid size.
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
        else:
            rank = self.embed_dim + self.mlp_dim
            layer1 = TP(
                input_size=self.tensor_input_size_layer1,
                output_size=self.mlp_dim,
                rank=rank,
                ignore_modes=ignore_modes,
                bias=bias,
            )

        if tensor_method_mlp[1] == "tdle":
            layer2 = TDLE(self.tensor_input_size_layer2, self.embed_dim, ignore_modes, bias, r=tdle_level)
        elif tensor_method_mlp[1] == "tle":
            layer2 = TLE(self.tensor_input_size_layer2, self.embed_dim, ignore_modes, bias)
        else:
            rank = self.mlp_dim + self.embed_dim
            layer2 = TP(
                input_size=self.tensor_input_size_layer2,
                output_size=self.embed_dim,
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
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, P_h+1, P_w, d1, d2, d3).

        Returns
        -------
        torch.Tensor
            Output of shape (B, P_h+1, P_w, d1, d2, d3).
        """
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _flops_tensor_layer(
    method: str,
    tensor_in: Tuple[int, int, int, int, int, int],
    out_dims: Tuple[int, int, int],
    ignore_modes: Iterable[int],
    bias: bool,
    tdle_level: int,
    in_dims: Optional[Tuple[int, int, int]] = None,
) -> int:
    """
    Analytic FLOPs for one tensorized mapping from in_dims to out_dims over `tensor_in`.

    For TP, provide `in_dims` explicitly; for TLE/TDLE it is derived from `tensor_in`.
    """
    B, P1, P2, d1, d2, d3 = tensor_in
    B_eff = B * P1 * P2

    if method == "tp":
        if in_dims is None:
            in_dims = (d1, d2, d3)
        rank = tuple(in_dims) + tuple(out_dims)
        parts = estimate_tp_flops(
            input_size=tensor_in,
            output_size=out_dims,
            rank=rank,
            ignore_modes=ignore_modes,
            include_bias=bias,
        )
        return int(parts["total"])

    if method == "tle":
        proj = tle_input_projector_flops(tensor_in, out_dims, ignore_modes)
        b = bias_add_flops(B_eff, out_dims) if bias else 0
        return int(proj + b)

    if method == "tdle":
        proj = tle_input_projector_flops(tensor_in, out_dims, ignore_modes)
        b = bias_add_flops(B_eff, out_dims) if bias else 0
        per = int(proj + b)
        sum_cost = (tdle_level - 1) * B_eff * math.prod(out_dims)
        return int(tdle_level * per + sum_cost)

    raise ValueError(f"Unknown tensor method: {method}")


def _sanity_check_once(
    input_size: Sequence[int],
    patch_size: int,
    embed_dim: Tuple[int, int, int],
    num_heads: Tuple[int, int, int],
    mlp_dim: Tuple[int, int, int],
    tensor_method: str = "tle",
    tensor_method_mlp: Tuple[str, str] = ("tle", "tle"),
    ignore_modes: Iterable[int] = (0, 1, 2),
    tdle_level: int = 2,
    bias: bool = True,
    out_embed: bool = True,
    include_softmax: bool = False,
    drop_path: float = 0.1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run shape, parameter counts (total and tensorized-only), timing, and analytic GFLOPs for one config.
    """
    device = device if device is not None else torch.device("cpu")
    enc = Encoder(
        input_size=input_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=0.1,
        bias=bias,
        out_embed=out_embed,
        drop_path=drop_path,
        ignore_modes=ignore_modes,
        tensor_method_mlp=tensor_method_mlp,
        tensor_method=tensor_method,
        tdle_level=tdle_level,
        reduce_level=(0, 0, 0),
    ).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(enc)

    attn = enc.attention
    attn_inner = 0
    for m in (attn.tensor_layer_Q, attn.tensor_layer_K, attn.tensor_layer_V):
        t, _ = param_counts(m)
        attn_inner += t
    if attn.tensor_layer_out is not None:
        t, _ = param_counts(attn.tensor_layer_out)
        attn_inner += t

    mlp_l1, mlp_l2 = enc.mlp[0], enc.mlp[3]
    mlp_inner = param_counts(mlp_l1)[0] + param_counts(mlp_l2)[0]

    print(
        f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] Params: total={total_params:,}, "
        f"trainable={trainable_params:,} | attention-only={attn_inner:,} | mlp-only={mlp_inner:,}"
    )

    B, C, H, W = map(int, input_size)
    P_h, P_w = H // patch_size, W // patch_size
    x = torch.randn(B, P_h + 1, P_w, *tuple(e - 0 for e in embed_dim), device=device, dtype=dtype, requires_grad=True)

    for _ in range(warmup):
        _ = enc(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y = enc(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    assert tuple(y.shape) == tuple(x.shape)

    y.mean().backward()
    assert x.grad is not None and tuple(x.grad.shape) == tuple(x.shape)

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] Input shape:  {tuple(x.shape)}")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] Output shape: {tuple(y.shape)}")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    d1, d2, d3 = enc.embed_dim
    seq = (P_h + 1) * P_w
    heads = num_heads
    h1, h2, h3 = heads
    dph = (d1 // h1) * (d2 // h2) * (d3 // h3)

    tensor_in = (B, P_h + 1, P_w, d1, d2, d3)

    fl_q = _flops_tensor_layer(tensor_method, tensor_in, (d1, d2, d3), ignore_modes, bias, tdle_level, in_dims=(d1, d2, d3))
    fl_k = fl_q
    fl_v = fl_q
    fl_qk = 2 * B * (h1 * h2 * h3) * seq * seq * dph
    fl_av = 2 * B * (h1 * h2 * h3) * seq * seq * dph
    fl_softmax = (2 * B * (h1 * h2 * h3) * seq * seq) if include_softmax else 0
    fl_out = _flops_tensor_layer(tensor_method, tensor_in, (d1, d2, d3), ignore_modes, bias, tdle_level, in_dims=(d1, d2, d3)) if attn.tensor_layer_out is not None else 0

    fl_ln = 2 * layernorm_flops(B, seq, d1 * d2 * d3)
    fl_res = 2 * residual_add_flops(B, seq, d1 * d2 * d3)
    fl_dp = (2 * droppath_flops(B, seq, d1 * d2 * d3)) if isinstance(enc.drop_path, DropPath) and enc.drop_path.drop_prob > 0 else 0

    mlp_l1_in = (B, P_h + 1, P_w, d1, d2, d3)
    mlp_l2_in = (B, P_h + 1, P_w, enc.mlp_dim[0], enc.mlp_dim[1], enc.mlp_dim[2])

    m1_method, m2_method = tensor_method_mlp
    fl_m1 = _flops_tensor_layer(m1_method, mlp_l1_in, enc.mlp_dim, ignore_modes, bias, tdle_level, in_dims=(d1, d2, d3))
    fl_m2 = _flops_tensor_layer(m2_method, mlp_l2_in, (d1, d2, d3), ignore_modes, bias, tdle_level, in_dims=enc.mlp_dim)

    total_flops = fl_ln + fl_q + fl_k + fl_v + fl_qk + fl_av + fl_softmax + fl_out + fl_m1 + fl_m2 + fl_res + fl_dp

    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs LayerNorm: {to_gflops(fl_ln):.3f} GFLOPs")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs Q/K/V:    {to_gflops(fl_q + fl_k + fl_v):.3f} GFLOPs")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs QK^T:     {to_gflops(fl_qk):.3f} GFLOPs")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs A·V:      {to_gflops(fl_av):.3f} GFLOPs")
    if fl_softmax:
        print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs Softmax: {to_gflops(fl_softmax):.3f} GFLOPs")
    if fl_out:
        print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs OutProj: {to_gflops(fl_out):.3f} GFLOPs")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs MLP-1:    {to_gflops(fl_m1):.3f} GFLOPs")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs MLP-2:    {to_gflops(fl_m2):.3f} GFLOPs")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs Residuals:{to_gflops(fl_res):.3f} GFLOPs")
    if fl_dp:
        print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs DropPath: {to_gflops(fl_dp):.3f} GFLOPs")
    print(f"[Enc/{tensor_method}|MLP:{tensor_method_mlp}] FLOPs TOTAL:    {to_gflops(total_flops):.3f} GFLOPs")


def sanity_check() -> None:
    """
    Run tensorized Encoder sanity checks across combinations.
    """
    B, C, H, W = 256, 3, 32, 32
    ps = 8
    embed_dim = (4, 4, 4)
    mlp_dim = (4, 4, 8)
    heads = (2, 2, 2)
    combos = [
        ("tle", ("tle", "tle"), True, True),
        ("tdle", ("tdle", "tdle"), True, True),
        ("tp", ("tp", "tp"), True, True),
        ("tle", ("tp", "tp"), True, False),
        ("tp", ("tle", "tle"), False, False),
    ]
    for method, mlp_methods, bias, out_embed in combos:
        _sanity_check_once(
            input_size=(B, C, H, W),
            patch_size=ps,
            embed_dim=embed_dim,
            num_heads=heads,
            mlp_dim=mlp_dim,
            tensor_method=method,
            tensor_method_mlp=mlp_methods,
            bias=bias,
            out_embed=out_embed,
            include_softmax=False,
            drop_path=0.1,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
