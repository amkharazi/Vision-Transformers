
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from einops import rearrange

from Tensorized_Layers.TCL import TCL
# from Tensorized_Layers.TRL import TRL

class PatchEmbedding(nn.Module):
    def __init__(self,input_size, patch_size, in_channels, embed_dim, device):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.tcl_input_size =  (input_size[0], input_size[1], input_size[2],
                                patch_size, patch_size, in_channels) # patched input image size
                                
        self.tcl = TCL(input_size=self.tcl_input_size,
                        rank=embed_dim,
                        ignore_modes=(0,1,2),
                        bias=True, 
                        device=device)

    def forward(self, x):
        x = rearrange(x, 
                        'b c (p1 h) (p2 w) -> b p1 p2 h w c',
                        h=self.patch_size, w=self.patch_size) # X = [B P1 P2 H W C]
        x = self.tcl(x) # X = [B P1 P2 D1 D2 D3]
        return x # patches