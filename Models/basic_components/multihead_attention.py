
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, out_embed = True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.out_embed = out_embed

        # first dim is the dimension of embedded x second dim is the embedded dimension of q/k/v (could be different)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # fc_out as the final embedding of the attention. commonly used
        if self.out_embed:
            self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        Q = rearrange(self.query(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.key(x), 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.value(x), 'b n (h d) -> b h n d', h=self.num_heads)

        # Attention calculation
        attn = torch.einsum('bhid,bhjd->bhij', Q, K) * self.scale

        attention = F.softmax(attn, dim=-1)
        x = torch.einsum('bhij,bhjd->bhid', attention, V)
        x = rearrange(x, 'b h n d -> b n (h d)')
        if self.out_embed:
            x = self.fc_out(x)
        return x