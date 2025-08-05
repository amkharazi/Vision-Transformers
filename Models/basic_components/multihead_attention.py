
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self,input_size, patch_size, embed_dim, num_heads, bias = True, out_embed = True):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.bias = bias

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
 
        self.out_embed = out_embed

        self.query = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=self.bias)
        self.key = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=self.bias)
        self.value = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=self.bias)

        if self.out_embed:
            self.fc_out = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=self.bias)

    def forward(self, x):

        Q = rearrange(self.query(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.key(x), 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.value(x), 'b n (h d) -> b h n d', h=self.num_heads)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention = F.softmax(attn, dim=-1)
        x = torch.matmul(attention, V)

        x = rearrange(x, 'b h n d -> b n (h d)')
        if self.out_embed:
            x = self.fc_out(x)
        return x