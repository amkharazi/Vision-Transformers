import sys
sys.path.append('.')

import torch
import torch.nn as nn
from models.basic_components.multihead_attention import MultiHeadAttention as MHA

class DropPath(nn.Module):
    """
    Stochastic depth per sample (batch-wise). No-op in eval mode.

    Args:
        drop_prob (float): Probability of dropping the residual branch ∈ [0, 1).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        assert isinstance(drop_prob, (int, float)), "drop_prob must be a number"
        assert 0.0 <= drop_prob < 1.0, "drop_prob must be in [0, 1)"
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

    Args:
        embed_dim (int): Token embedding size D.
        num_heads (int): Attention heads H.
        mlp_dim (int): Hidden size of the MLP.
        dropout (float): Dropout probability inside the MLP.
        bias (bool): Use bias in linear layers.
        out_embed (bool): Apply output projection in MHA.
        drop_path (float): Stochastic depth rate.
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

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.attention = MHA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            out_proj=out_embed,   # matches the cleaned MHA API
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
        Args:
            x (Tensor): Input of shape (B, N, D).

        Returns:
            Tensor: Output of shape (B, N, D).
        """
        assert x.dim() == 3, f"Expected (B, N, D), got {tuple(x.shape)}"
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == "__main__":
    # sanity check
    torch.manual_seed(0)
    B, N, D, H = 2, 196, 768, 12
    mlp_dim = 4 * D

    x = torch.randn(B, N, D, requires_grad=True)
    enc = Encoder(
        embed_dim=D,
        num_heads=H,
        mlp_dim=mlp_dim,
        dropout=0.1,
        drop_path=0.2,
    )
    enc.train()
    y = enc(x)
    assert y.shape == (B, N, D)

    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape

    enc.eval()
    y2 = enc(x.detach())
    assert y2.shape == (B, N, D)

    print("Encoder sanity check passed.")
