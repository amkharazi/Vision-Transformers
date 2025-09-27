import sys
sys.path.append('.')

import time
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.num_param import param_counts
from utils.flops import mha_flops, to_gflops


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention on sequences of shape (B, N, D).

    Parameters
    ----------
    embed_dim : int
        Model dimension D.
    num_heads : int
        Number of heads H (D must be divisible by H).
    bias : bool, default=True
        Use bias in Q/K/V and output projections.
    out_proj : bool, default=True
        Apply final linear projection.
    attn_dropout : float, default=0.0
        Dropout on attention weights.
    proj_dropout : float, default=0.0
        Dropout after output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        out_proj: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert isinstance(embed_dim, int) and embed_dim > 0
        assert isinstance(num_heads, int) and num_heads > 0
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj_enabled = out_proj
        if out_proj:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Union[torch.Tensor, None] = None,
        key_padding_mask: Union[torch.Tensor, None] = None,
        need_weights: bool = False,
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, N, D).
        attn_mask : torch.Tensor | None
            Bool or additive mask, broadcastable to (B, H, N, N).
        key_padding_mask : torch.Tensor | None
            Bool mask of shape (B, N) where True marks padded tokens.
        need_weights : bool, default=False
            If True, also returns averaged attention weights of shape (B, N, N).

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Output y of shape (B, N, D), and optionally attention weights.
        """
        assert x.dim() == 3, f"Expected (B, N, D), got {tuple(x.shape)}"
        B, N, D = x.shape
        assert D == self.embed_dim

        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.v(x), "b n (h d) -> b h n d", h=self.num_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        y = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")

        if self.out_proj_enabled:
            y = self.out_proj(y)
            y = self.proj_drop(y)

        if need_weights:
            return y, attn.mean(dim=1)
        return y


def _sanity_check_once(
    B: int,
    N: int,
    D: int,
    H: int,
    bias: bool = True,
    out_proj: bool = True,
    attn_dropout: float = 0.0,
    proj_dropout: float = 0.0,
    include_softmax: bool = False,
    need_weights: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run shape, parameter-count, timing, and analytic FLOPs check for MHA.

    Parameters
    ----------
    B, N, D, H : int
        Batch, tokens, embed dim, heads.
    bias : bool, default=True
        Bias in linear layers.
    out_proj : bool, default=True
        Use output projection.
    attn_dropout : float, default=0.0
        Dropout on attention map.
    proj_dropout : float, default=0.0
        Dropout after output projection.
    include_softmax : bool, default=False
        Include softmax ops in analytic FLOPs.
    need_weights : bool, default=True
        Also fetch attention weights for validation.
    device : torch.device, optional
        Device to use.
    dtype : torch.dtype, default=torch.float32
        Tensor dtype.
    warmup : int, default=2
        Warmup forward passes.
    iters : int, default=5
        Timed forward passes to average.
    """
    device = device if device is not None else torch.device("cpu")
    x = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=True)

    mha = MultiHeadAttention(
        embed_dim=D,
        num_heads=H,
        bias=bias,
        out_proj=out_proj,
        attn_dropout=attn_dropout,
        proj_dropout=proj_dropout,
    ).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(mha)
    print(f"[MHA] Parameters: total={total_params:,}, trainable={trainable_params:,}")

    for _ in range(warmup):
        _ = mha(x, need_weights=need_weights)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = mha(x, need_weights=need_weights)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    if need_weights:
        y, w = out  # type: ignore[assignment]
        assert tuple(y.shape) == (B, N, D)
        assert tuple(w.shape) == (B, N, N)
        assert torch.allclose(w.sum(dim=-1), torch.ones(B, N, device=w.device, dtype=w.dtype), atol=1e-5)
    else:
        y = out  # type: ignore[assignment]
        assert tuple(y.shape) == (B, N, D)

    y.sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == (B, N, D)

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[MHA] Input shape:  {(B, N, D)}")
    print(f"[MHA] Output shape: {(B, N, D)}")
    print(f"[MHA] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    fl = mha_flops(
        batch=B,
        tokens=N,
        embed_dim=D,
        num_heads=H,
        out_proj=out_proj,
        include_bias=bias,
        include_softmax=include_softmax,
    )
    print(f"[MHA] FLOPs QKV:     {to_gflops(fl['qkv']):.3f} GFLOPs")
    print(f"[MHA] FLOPs QK^T:    {to_gflops(fl['qkT']):.3f} GFLOPs")
    print(f"[MHA] FLOPs A·V:     {to_gflops(fl['attnV']):.3f} GFLOPs")
    print(f"[MHA] FLOPs Proj:    {to_gflops(fl['proj']):.3f} GFLOPs")
    if 'softmax' in fl and fl['softmax'] > 0:
        print(f"[MHA] FLOPs Softmax: {to_gflops(fl['softmax']):.3f} GFLOPs")
    print(f"[MHA] FLOPs TOTAL:   {to_gflops(fl['total']):.3f} GFLOPs")


def sanity_check() -> None:
    """
    Run MHA sanity check with a ViT-like configuration.
    """
    B, N, D, H = 2, 196, 768, 12
    _sanity_check_once(B, N, D, H, bias=True, out_proj=True, include_softmax=False, need_weights=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
