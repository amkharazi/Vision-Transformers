import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer.

    Splits an image into non-overlapping patches and projects each patch to a fixed
    embedding size, returning shape (B, N, D) where N = (H/ps)*(W/ps).

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        patch_size (int): Patch size (kernel_size=stride=patch_size).
        embed_dim (int): Output embedding dimension per patch.
        bias (bool): Use bias in the projection layer.
        use_conv (bool): If True, uses Conv2d; otherwise uses Linear on flattened patches.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        bias: bool = True,
        use_conv: bool = True,
    ):
        super().__init__()

        # basic validations
        assert (
            isinstance(in_channels, int) and in_channels > 0
        ), "in_channels must be a positive int"
        assert (
            isinstance(patch_size, int) and patch_size > 0
        ), "patch_size must be a positive int"
        assert (
            isinstance(embed_dim, int) and embed_dim > 0
        ), "embed_dim must be a positive int"

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_conv = use_conv

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
            self.projection = nn.Linear(
                in_features=in_features, out_features=embed_dim, bias=bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, N, D) with N = (H/ps)*(W/ps), D = embed_dim.
        """
        assert x.dim() == 4, f"Expected 4D input (B, C, H, W), got {x.dim()}D"
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert (
            H % self.patch_size == 0 and W % self.patch_size == 0
        ), f"Input H and W must be divisible by patch_size={self.patch_size}"

        if self.use_conv:
            x = self.projection(x)  # (B, D, H/ps, W/ps)
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        else:
            x = rearrange(
                x,
                "b c (p1 h) (p2 w) -> b (p1 p2) (c h w)",
                h=self.patch_size,
                w=self.patch_size,
            )  # (B, N, C*ps*ps)
            x = self.projection(x)  # (B, N, D)

        return x


if __name__ == "__main__":
    # sanity check
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 224, 224
    ps, D = 16, 768

    x = torch.randn(B, C, H, W, requires_grad=True)

    # conv variant
    pe_conv = PatchEmbedding(in_channels=C, patch_size=ps, embed_dim=D, use_conv=True)
    out_conv = pe_conv(x)
    assert out_conv.shape == (B, (H // ps) * (W // ps), D)

    # linear variant
    pe_lin = PatchEmbedding(in_channels=C, patch_size=ps, embed_dim=D, use_conv=False)
    out_lin = pe_lin(x.detach())  # no grad flow needed for this check
    assert out_lin.shape == (B, (H // ps) * (W // ps), D)

    # gradient flows
    out_conv.mean().backward()
    assert x.grad is not None and x.grad.shape == x.shape

    print("PatchEmbedding sanity check passed.")
