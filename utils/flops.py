"""
Analytic FLOPs estimators for tensorized models.

Assumptions
-----------
1) The first dimension of `input_size` is batch B, e.g. (B, C, H, W).
2) TLE performs sequential mode-n projections along the active modes, in order,
   each as a matrix multiply that changes mode size d_in -> d_out.
   Cost per mode is 2 * d_in * d_out * prod(other current dims).
3) The TP forward contracts x_hat (B, *input_rank) with g_hat (*input_rank, *output_size)
   via a tensordot over all `input_rank` modes:
   FLOPs = 2 * B * prod(input_rank) * prod(output_size).
4) Bias add, if present: FLOPs = B * prod(output_size).
"""

from typing import Dict, Sequence, Iterable, Optional, Tuple
import math
import torch.nn as nn
import torch

def _prod(x: Sequence[int]) -> int:
    """Integer product."""
    return int(math.prod(int(v) for v in x)) if len(x) else 1


def _sequential_projection_flops(
    base_shape: Sequence[int],
    active_modes: Sequence[int],
    target_dims: Sequence[int],
) -> int:
    """
    FLOPs for sequential mode-n projections.

    Parameters
    ----------
    base_shape : Sequence[int]
        Working tensor shape before projections. Includes batch if present.
    active_modes : Sequence[int]
        Indices of modes to project, in the order they are applied.
    target_dims : Sequence[int]
        Target size for each active mode, same length/order as `active_modes`.

    Returns
    -------
    int
        Total FLOPs for the full sequence of projections.
    """
    shape = list(int(v) for v in base_shape)
    total = 0
    for mode, d_out in zip(active_modes, target_dims):
        d_in = int(shape[mode])
        outer = _prod(shape) // d_in
        total += 2 * d_in * int(d_out) * outer
        shape[mode] = int(d_out)
    return int(total)


def tensordot_flops(batch: int, input_rank: Sequence[int], output_size: Sequence[int]) -> int:
    """
    FLOPs for contracting (B, *input_rank) with (*input_rank, *output_size).

    Parameters
    ----------
    batch : int
        Batch size B.
    input_rank : Sequence[int]
        Contracted rank sizes.
    output_size : Sequence[int]
        Output spatial/channel sizes.

    Returns
    -------
    int
        FLOPs for the contraction.
    """
    return int(2 * int(batch) * _prod(input_rank) * _prod(output_size))


def bias_add_flops(batch: int, output_size: Sequence[int]) -> int:
    """
    FLOPs for adding a bias of shape `output_size` to each item in a batch.

    Parameters
    ----------
    batch : int
        Batch size B.
    output_size : Sequence[int]
        Output spatial/channel sizes.

    Returns
    -------
    int
        FLOPs for the bias addition.
    """
    return int(int(batch) * _prod(output_size))


def tle_input_projector_flops(
    input_size: Sequence[int],
    input_rank: Sequence[int],
    ignore_modes: Iterable[int],
) -> int:
    """
    FLOPs for projecting input tensor to input ranks.

    Parameters
    ----------
    input_size : Sequence[int]
        Full input shape including batch, e.g. (B, C, H, W).
    input_rank : Sequence[int]
        Target ranks for active modes.
    ignore_modes : Iterable[int]
        Modes not projected, e.g. (0,) to ignore batch.

    Returns
    -------
    int
        FLOPs for the input projector.
    """
    L = len(input_size)
    ign = set(int(i) for i in ignore_modes)
    active = [i for i in range(L) if i not in ign]
    return _sequential_projection_flops(input_size, active, input_rank)


def tle_core_projector_flops(
    input_rank: Sequence[int],
    output_rank: Sequence[int],
    output_size: Sequence[int],
) -> int:
    """
    FLOPs for projecting the core tensor to `output_size`.

    Parameters
    ----------
    input_rank : Sequence[int]
        Input-side ranks (not projected here).
    output_rank : Sequence[int]
        Output-side ranks to be projected.
    output_size : Sequence[int]
        Target output sizes.

    Returns
    -------
    int
        FLOPs for the core projector.
    """
    base = tuple(int(v) for v in list(input_rank) + list(output_rank))
    offset = len(input_rank)
    active = [offset + i for i in range(len(output_rank))]
    return _sequential_projection_flops(base, active, output_size)


def estimate_tp_flops(
    input_size: Sequence[int],
    output_size: Sequence[int],
    rank: Sequence[int],
    ignore_modes: Iterable[int] = (0,),
    include_bias: bool = True,
) -> Dict[str, int]:
    """
    Analytic FLOPs breakdown for the TP layer.

    Parameters
    ----------
    input_size : Sequence[int]
        Full input shape including batch.
    output_size : Sequence[int]
        Output shape excluding batch.
    rank : Sequence[int]
        Combined ranks; first n go to input, last m to output.
    ignore_modes : Iterable[int], default=(0,)
        Modes ignored in the input projector.
    include_bias : bool, default=True
        Whether to include bias add FLOPs.

    Returns
    -------
    Dict[str, int]
        Keys: 'input_projector', 'tensordot', 'core_projector', 'bias', 'total'.
    """
    input_size = tuple(int(v) for v in input_size)
    output_size = tuple(int(v) for v in output_size)
    rank = tuple(int(v) for v in rank)

    n = len(input_size) - len(tuple(ignore_modes))
    m = len(output_size)
    input_rank = rank[:n]
    output_rank = rank[n:]

    fp_in = tle_input_projector_flops(input_size, input_rank, ignore_modes)
    fp_dot = tensordot_flops(input_size[0], input_rank, output_size)
    fp_core = tle_core_projector_flops(input_rank, output_rank, output_size)
    fp_bias = bias_add_flops(input_size[0], output_size) if include_bias else 0

    total = fp_in + fp_dot + fp_core + fp_bias
    return {
        "input_projector": int(fp_in),
        "tensordot": int(fp_dot),
        "core_projector": int(fp_core),
        "bias": int(fp_bias),
        "total": int(total),
    }


def to_gflops(flops: int) -> float:
    """
    Convert FLOPs to GFLOPs.

    Parameters
    ----------
    flops : int
        Floating-point operation count.

    Returns
    -------
    float
        GFLOPs.
    """
    return float(flops) / 1e9

def linear_flops(batch: int, in_features: int, out_features: int, include_bias: bool = True) -> int:
    """
    FLOPs for y = x @ W^T (+ b) where x is (B, in_features) and y is (B, out_features).

    Parameters
    ----------
    batch : int
        Batch size B.
    in_features : int
    out_features : int
    include_bias : bool, default=True
        Whether to include B * out_features bias adds.

    Returns
    -------
    int
        Floating-point operation count.
    """
    macs = 2 * int(batch) * int(in_features) * int(out_features)
    bias = int(batch) * int(out_features) if include_bias else 0
    return int(macs + bias)

def layernorm_flops(batch: int, tokens: int, dim: int) -> int:
    """
    Approx FLOPs for LayerNorm on (B, N, D): ~5 * B * N * D.
    """
    return int(5 * int(batch) * int(tokens) * int(dim))

def residual_add_flops(batch: int, tokens: int, dim: int) -> int:
    """
    FLOPs for residual add on (B, N, D): B * N * D.
    """
    return int(int(batch) * int(tokens) * int(dim))

def droppath_flops(batch: int, tokens: int, dim: int) -> int:
    """
    FLOPs for DropPath mask+scale on (B, N, D): ~B * N * D.
    """
    return int(int(batch) * int(tokens) * int(dim))

def mha_flops(
    batch: int,
    tokens: int,
    embed_dim: int,
    num_heads: int,
    out_proj: bool = True,
    include_bias: bool = True,
    include_softmax: bool = False,
) -> Dict[str, int]:
    """
    Analytic FLOPs for multi-head self-attention on (B, N, D).
    """
    assert embed_dim % num_heads == 0
    d = embed_dim // num_heads
    B, N, D, H = int(batch), int(tokens), int(embed_dim), int(num_heads)

    qkv = 3 * linear_flops(B * N, D, D, include_bias=include_bias)
    qk = 2 * B * H * N * N * d
    av = 2 * B * H * N * N * d
    proj = linear_flops(B * N, D, D, include_bias=include_bias) if out_proj else 0
    softmax = (2 * B * H * N * N) if include_softmax else 0

    total = qkv + qk + av + proj + softmax
    return {"qkv": int(qkv), "qkT": int(qk), "attnV": int(av), "proj": int(proj), "softmax": int(softmax), "total": int(total)}

def mlp_flops(batch: int, tokens: int, dim_in: int, dim_hidden: int, include_bias: bool = True) -> Dict[str, int]:
    """
    Analytic FLOPs for 2-layer MLP with activation, on (B, N, D).
    """
    B, N, D, H = int(batch), int(tokens), int(dim_in), int(dim_hidden)
    up = linear_flops(B * N, D, H, include_bias=include_bias)
    down = linear_flops(B * N, H, D, include_bias=include_bias)
    total = up + down
    return {"up": int(up), "down": int(down), "total": int(total)}

def encoder_block_flops(
    batch: int,
    tokens: int,
    dim: int,
    num_heads: int,
    mlp_dim: int,
    out_proj: bool = True,
    include_bias: bool = True,
    include_layernorm: bool = True,
    include_residual: bool = True,
    drop_path_rate: float = 0.0,
) -> Dict[str, int]:
    """
    Analytic FLOPs for a pre-norm Transformer encoder block.
    """
    B, N, D = int(batch), int(tokens), int(dim)

    ln = 2 * layernorm_flops(B, N, D) if include_layernorm else 0
    att = mha_flops(B, N, D, num_heads, out_proj=out_proj, include_bias=include_bias)
    mlp = mlp_flops(B, N, D, mlp_dim, include_bias=include_bias)
    res = 2 * residual_add_flops(B, N, D) if include_residual else 0
    dp = (2 * droppath_flops(B, N, D)) if drop_path_rate > 0.0 else 0

    total = ln + att["total"] + mlp["total"] + res + dp
    return {
        "layernorm": int(ln),
        "mha_total": int(att["total"]),
        "mlp_total": int(mlp["total"]),
        "residuals": int(res),
        "droppath": int(dp),
        "total": int(total),
        **{f"mha_{k}": v for k, v in att.items()},
        **{f"mlp_{k}": v for k, v in mlp.items()},
    }


def conv2d_flops(
    batch: int,
    in_channels: int,
    out_channels: int,
    in_h: int,
    in_w: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    include_bias: bool = True,
) -> int:
    """
    FLOPs for a standard 2D convolution on (B, C_in, H, W).

    Returns
    -------
    int
        Floating-point operation count; multiply-add counts as 2 FLOPs.
    """
    B = int(batch); C_in = int(in_channels); C_out = int(out_channels)
    H = int(in_h); W = int(in_w); K = int(kernel_size); S = int(stride); P = int(padding); G = int(groups)
    H_out = (H + 2 * P - K) // S + 1
    W_out = (W + 2 * P - K) // S + 1
    macs_per_out = 2 * (C_in // G) * K * K
    conv = B * C_out * H_out * W_out * macs_per_out
    bias = (B * C_out * H_out * W_out) if include_bias else 0
    return int(conv + bias)

def conv_patch_embed_flops(
    batch: int,
    in_channels: int,
    in_h: int,
    in_w: int,
    patch_size: int,
    embed_dim: int,
    include_bias: bool = True,
) -> int:
    """
    FLOPs for Conv2d patch embedding with kernel=stride=patch_size, padding=0.
    """
    return conv2d_flops(
        batch=batch,
        in_channels=in_channels,
        out_channels=int(embed_dim),
        in_h=in_h,
        in_w=in_w,
        kernel_size=int(patch_size),
        stride=int(patch_size),
        padding=0,
        groups=1,
        include_bias=include_bias,
    )


def try_thop_gflops(model: nn.Module, input_size: Tuple[int, int, int, int]) -> Optional[float]:
    try:
        from thop import profile
    except Exception:
        return None

    def _profile(m: nn.Module, dev: torch.device) -> Optional[float]:
        was_training = m.training
        m.eval()
        try:
            dummy = torch.empty(*input_size, device=dev)
            with torch.no_grad():
                flops, _ = profile(m, inputs=(dummy,), verbose=False)
            return float(flops) / 1e9
        except Exception:
            return None
        finally:
            if was_training:
                m.train()

    dev = next(model.parameters()).device
    g = _profile(model, dev)
    if g is not None:
        return g

    try:
        model_cpu = model.to("cpu")
        g = _profile(model_cpu, torch.device("cpu"))
        model.to(dev)
        return g
    except Exception:
        try:
            model.to(dev)
        except Exception:
            pass
        return None
