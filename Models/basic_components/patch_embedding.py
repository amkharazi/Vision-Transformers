
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        x = rearrange(x,
                        'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)',
                        h=self.patch_size, w=self.patch_size)
        x = self.projection(x) # X = [B, N, D]
        return x # patches