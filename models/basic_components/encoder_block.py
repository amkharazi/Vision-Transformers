import sys
sys.path.append('.')

import time
from typing import Optional

import torch
import torch.nn as nn
from models.basic_components.multihead_attention import MultiHeadAttention as MHA

from utils.num_param import param_counts
from utils.flops import encoder_block_flops, to_gflops


class DropPath(nn.Module):
    """
    Stochastic depth per sample (batch-wise). No-op in eval mode.

    Parameters
    ----------
    drop_prob : float
        Probability of dropping the residual branch ∈ [0, 1).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        assert isinstance(drop_prob, (int, float))
        assert 0.0 <= drop_prob < 1.0
        self.drop_prob = float(drop_prob)

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
    Pre-norm Transformer encoder block: MHA + MLP with residuals and optional DropPath.

    Parameters
    ----------
    embed_dim : int
        Token embedding size D.
    num_heads : int
        Attention heads H.
    mlp_dim : int
        Hidden size of the MLP.
    dropout : float
        Dropout probability inside the MLP.
    bias : bool
        Use bias in linear layers.
    out_embed : bool
        Apply output projection in MHA.
    drop_path : float
        Stochastic depth rate.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.5,
        bias: bool = True,
        out_embed: bool = True,
        drop_path: float = 0.1,
    ):
        super().__init__()
        assert isinstance(embed_dim, int) and embed_dim > 0
        assert isinstance(num_heads, int) and num_heads > 0
        assert isinstance(mlp_dim, int) and mlp_dim > 0
        assert 0.0 <= dropout < 1.0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.bias = bias
        self.out_embed = out_embed
        self.drop_path_rate = float(drop_path)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.attention = MHA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            out_proj=out_embed,
            attn_dropout=0.0,
            proj_dropout=0.0,
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, N, D).

        Returns
        -------
        torch.Tensor
            Output of shape (B, N, D).
        """
        assert x.dim() == 3, f"Expected (B, N, D), got {tuple(x.shape)}"
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _sanity_check_once(
    B: int,
    N: int,
    D: int,
    H: int,
    mlp_dim: int,
    dropout: float = 0.1,
    drop_path: float = 0.2,
    bias: bool = True,
    out_embed: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run shape, parameter-count, timing, and analytic FLOPs check for one Encoder block.

    Parameters
    ----------
    B : int
        Batch size.
    N : int
        Number of tokens.
    D : int
        Embedding dim.
    H : int
        Number of heads.
    mlp_dim : int
        MLP hidden dim.
    dropout : float
        MLP dropout.
    drop_path : float
        Stochastic depth rate.
    bias : bool
        Bias in linear layers.
    out_embed : bool
        Use output projection in MHA.
    device : torch.device, optional
        Device to use.
    dtype : torch.dtype
        Tensor dtype.
    warmup : int
        Warmup forward passes.
    iters : int
        Timed forward passes to average.
    """
    device = device if device is not None else torch.device("cpu")
    x = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=False)

    enc = Encoder(
        embed_dim=D,
        num_heads=H,
        mlp_dim=mlp_dim,
        dropout=dropout,
        bias=bias,
        out_embed=out_embed,
        drop_path=drop_path,
    ).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(enc)
    print(f"[Encoder] Parameters: total={total_params:,}, trainable={trainable_params:,}")

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

    assert tuple(y.shape) == (B, N, D)

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[Encoder] Input shape:  {(B, N, D)}")
    print(f"[Encoder] Output shape: {(B, N, D)}")
    print(f"[Encoder] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    fl = encoder_block_flops(
        batch=B,
        tokens=N,
        dim=D,
        num_heads=H,
        mlp_dim=mlp_dim,
        out_proj=out_embed,
        include_bias=bias,
        include_layernorm=True,
        include_residual=True,
        drop_path_rate=drop_path,
    )

    print(f"[Encoder] FLOPs LayerNorm:   {to_gflops(fl['layernorm']):.3f} GFLOPs")
    print(f"[Encoder] FLOPs MHA (qkv):   {to_gflops(fl['mha_qkv']):.3f} GFLOPs")
    print(f"[Encoder] FLOPs MHA (QK^T):  {to_gflops(fl['mha_qkT']):.3f} GFLOPs")
    print(f"[Encoder] FLOPs MHA (AV):    {to_gflops(fl['mha_attnV']):.3f} GFLOPs")
    print(f"[Encoder] FLOPs MHA (proj):  {to_gflops(fl['mha_proj']):.3f} GFLOPs")
    print(f"[Encoder] FLOPs MLP:         {to_gflops(fl['mlp_total']):.3f} GFLOPs")
    print(f"[Encoder] FLOPs residuals:   {to_gflops(fl['residuals']):.3f} GFLOPs")
    if drop_path > 0.0:
        print(f"[Encoder] FLOPs DropPath:    {to_gflops(fl['droppath']):.3f} GFLOPs")
    print(f"[Encoder] FLOPs TOTAL:       {to_gflops(fl['total']):.3f} GFLOPs")


def sanity_check() -> None:
    """
    Run Encoder sanity check with a ViT-like configuration.
    """
    B, N, D, H = 2, 196, 768, 12
    mlp_dim = 4 * D
    _sanity_check_once(B, N, D, H, mlp_dim, dropout=0.1, drop_path=0.2, bias=True, out_embed=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
