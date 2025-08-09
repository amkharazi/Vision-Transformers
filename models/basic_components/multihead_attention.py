import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention on sequences of shape (B, N, D).

    Args:
        embed_dim (int): Model dimension D.
        num_heads (int): Number of heads H (D must be divisible by H).
        bias (bool): Use bias in Q/K/V and output projections.
        out_proj (bool): Apply final linear projection.
        attn_dropout (float): Dropout on attention weights.
        proj_dropout (float): Dropout after output projection.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        out_proj: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(embed_dim, int) and embed_dim > 0
        assert isinstance(num_heads, int) and num_heads > 0
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

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
        
        
        attn_mask: Union[torch.Tensor, None] = None,        # (N, N) or (B, 1, N, N) or broadcastable
        key_padding_mask: Union[torch.Tensor, None] = None,   # (B, N) bool, True = pad
        need_weights: bool = False,
    ):
        """
        Args:
            x: (B, N, D)
            attn_mask: additive mask with -inf for disallowed positions or bool mask.
            key_padding_mask: bool mask of shape (B, N), True marks padded tokens.
            need_weights: if True, returns average attention weights over heads (B, N, N).

        Returns:
            y: (B, N, D) and optionally attn_weights if need_weights.
        """
        assert x.dim() == 3, f"Expected (B, N, D), got {tuple(x.shape)}"
        B, N, D = x.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.v(x), "b n (h d) -> b h n d", h=self.num_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if attn_mask is not None:
            # support bool or additive masks; broadcast to (B, H, N, N) if needed
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        if key_padding_mask is not None:
            # expand to (B, 1, 1, N) -> mask keys
            mask = key_padding_mask[:, None, None, :]  # bool
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)                      # (B, H, N, head_dim)
        y = rearrange(y, "b h n d -> b n (h d)")       # (B, N, D)

        if self.out_proj_enabled:
            y = self.out_proj(y)
            y = self.proj_drop(y)

        if need_weights:
            # average over heads for convenience
            return y, attn.mean(dim=1)
        return y


if __name__ == "__main__":
    # sanity check
    torch.manual_seed(0)
    B, N, D, H = 2, 196, 768, 12
    x = torch.randn(B, N, D, requires_grad=True)

    mha = MultiHeadAttention(embed_dim=D, num_heads=H, attn_dropout=0.0, proj_dropout=0.0)
    y, w = mha(x, need_weights=True)
    assert y.shape == (B, N, D)
    assert w.shape == (B, N, N)
    # rows of attention should sum to ~1
    assert torch.allclose(w.sum(dim=-1), torch.ones(B, N), atol=1e-5)

    # grad flows
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape

    print("MultiHeadAttention sanity check passed.")
