import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dims = embed_dim
        self.in_channels = in_channels

        self.w_h = nn.Parameter(torch.randn(self.patch_size, self.embed_dims[0]), requires_grad=True)
        self.w_w = nn.Parameter(torch.randn(self.patch_size, self.embed_dims[1]), requires_grad=True)
        self.w_c = nn.Parameter(torch.randn(self.in_channels, self.embed_dims[2]), requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 
                        'b c (p1 h) (p2 w) -> b p1 p2 h w c',
                        h=self.patch_size, w=self.patch_size) # X = [B P1 P2 H W C]
        x = torch.einsum(
                        'b p q h w c , h x , w y , c z  -> b p q x y z',
                        (x, self.w_h, self.w_w, self.w_c)
                        ) # X = [B P1 P2 D1 D2 D3]
        return x # patches