
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, bias = True, device = 'cuda', ignore_modes = None):
        super(PatchEmbedding, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.bias = bias
        self.device = device
        self.ignore_modes = None
        self.projection = nn.Linear(in_features=self.patch_size * self.patch_size * self.input_size[1],
                                     out_features=self.embed_dim,
                                     bias=self.bias,
                                     device=self.device)
    def forward(self, x):
        x = rearrange(x,
                        'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)',
                        h=self.patch_size, w=self.patch_size)
        x = self.projection(x) # X = [B, N, D]
        return x # patches