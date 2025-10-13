import sys
sys.path.append('.')

import time
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from utils.num_param import param_counts
from utils.flops import linear_flops, conv_patch_embed_flops, to_gflops


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer.

    Splits an image into non-overlapping patches and projects each patch to a fixed
    embedding size, returning shape (B, N, D) where N = (H/ps) * (W/ps).

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB).
    patch_size : int
        Patch size; uses kernel_size=stride=patch_size.
    embed_dim : int
        Output embedding dimension per patch.
    bias : bool, default=True
        Use bias in the projection layer.
    use_conv : bool, default=True
        If True, uses Conv2d; otherwise uses Linear on flattened patches.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        bias: bool = True,
        use_conv: bool = True,
    ) -> None:
        super().__init__()

        assert isinstance(in_channels, int) and in_channels > 0
        assert isinstance(patch_size, int) and patch_size > 0
        assert isinstance(embed_dim, int) and embed_dim > 0

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_conv = use_conv
        self.bias_flag = bool(bias)

        if use_conv:
            self.projection = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        else:
            in_features = patch_size * patch_size * in_channels
            self.projection = nn.Linear(in_features=in_features, out_features=embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output of shape (B, N, D) with N = (H/ps)*(W/ps) and D = embed_dim.
        """
        assert x.dim() == 4, f"Expected 4D input (B, C, H, W), got {x.dim()}D"
        B, C, H, W = x.shape
        assert C == self.in_channels
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        if self.use_conv:
            return self.projection(x).flatten(2).transpose(1, 2)
        return self.projection(
            rearrange(
                x,
                "b c (p1 h) (p2 w) -> b (p1 p2) (c h w)",
                h=self.patch_size,
                w=self.patch_size,
            )
        )


def _sanity_check_once(
    B: int,
    C: int,
    H: int,
    W: int,
    patch_size: int,
    embed_dim: int,
    use_conv: bool,
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run shape, parameter-count, timing, and analytic FLOPs check for one PatchEmbedding variant.

    Parameters
    ----------
    B, C, H, W : int
        Input tensor dimensions.
    patch_size : int
        Patch size.
    embed_dim : int
        Embedding dimension.
    use_conv : bool
        True for Conv2d; False for Linear.
    bias : bool, default=True
        Bias in the projection layer.
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
    x = torch.randn(B, C, H, W, device=device, dtype=dtype, requires_grad=False)
    model = PatchEmbedding(in_channels=C, patch_size=patch_size, embed_dim=embed_dim, bias=bias, use_conv=use_conv).to(
        device=device, dtype=dtype
    )

    total_params, trainable_params = param_counts(model)
    tag = "Conv2d" if use_conv else "Linear"
    print(f"[PatchEmbedding/{tag}] Parameters: total={total_params:,}, trainable={trainable_params:,}")

    for _ in range(warmup):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    N = (H // patch_size) * (W // patch_size)
    assert tuple(y.shape) == (B, N, embed_dim)

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[PatchEmbedding/{tag}] Input shape:  {(B, C, H, W)}")
    print(f"[PatchEmbedding/{tag}] Output shape: {(B, N, embed_dim)}")
    print(f"[PatchEmbedding/{tag}] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    if use_conv:
        fl = conv_patch_embed_flops(B, C, H, W, patch_size, embed_dim, include_bias=bias)
    else:
        fl = linear_flops(B * N, C * patch_size * patch_size, embed_dim, include_bias=bias)
    print(f"[PatchEmbedding/{tag}] FLOPs TOTAL: {to_gflops(fl):.3f} GFLOPs")


def sanity_check() -> None:
    """
    Run PatchEmbedding sanity checks for Conv2d and Linear variants.
    """
    B, C, H, W = 2, 3, 224, 224
    ps, D = 16, 768
    _sanity_check_once(B, C, H, W, ps, D, use_conv=True, bias=True)
    _sanity_check_once(B, C, H, W, ps, D, use_conv=False, bias=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
