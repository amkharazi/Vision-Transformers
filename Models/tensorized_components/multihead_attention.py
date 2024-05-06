
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from Tensorized_Layers.TCL import TCL
# from Tensorized_Layers.TRL import TRL

class MultiHeadAttention(nn.Module):
    def __init__(self,input_size, patch_size, embed_dim, num_heads, bias = True, out_embed = True, device = 'cuda', ignore_modes = (0,1,2)):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        self.bias = bias
        self.ignore_modes = ignore_modes

        self.h1 = num_heads[0]
        self.h2 = num_heads[1]
        self.h3 = num_heads[2]
        
        self.scale = ((self.embed_dim[0]//self.h1)*
                      (self.embed_dim[1]//self.h2)*
                      (self.embed_dim[2]//self.h3))** -0.5
        
        self.out_embed = out_embed

        self.tcl_input_size =  (self.input_size[0], self.input_size[2]//self.patch_size + 1, self.input_size[3]//self.patch_size,
                                self.embed_dim[0], self.embed_dim[1], self.embed_dim[2]) # patched input image size


        # first dim is the dimension of embedded x second dim is the embedded dimension of q/k/v (could be different)
        #  Q
        self.tcl_q = TCL(input_size=self.tcl_input_size,
                        rank=self.embed_dim,
                        ignore_modes=self.ignore_modes,
                        bias=self.bias, 
                        device=self.device)
        # K
        self.tcl_k = TCL(input_size=self.tcl_input_size,
                        rank=self.embed_dim,
                        ignore_modes=self.ignore_modes,
                        bias=self.bias, 
                        device=self.device)

        # V
        self.tcl_v = TCL(input_size=self.tcl_input_size,
                        rank=self.embed_dim,
                        ignore_modes=self.ignore_modes,
                        bias=self.bias, 
                        device=self.device)
        if self.out_embed:
            self.tcl_out = TCL(input_size=self.tcl_input_size,
                        rank=self.embed_dim,
                        ignore_modes=self.ignore_modes,
                        bias=self.bias, 
                        device=self.device)

    def forward(self, x):
        q = self.tcl_q(x)
        k = self.tcl_k(x)
        v = self.tcl_v(x)

        Q = rearrange(q, 'b p1 p2 (x h1) (y h2) (z h3) -> b p1 p2 h1 h2 h3 x y z', h1 = self.h1, h2 = self.h2, h3 = self.h3)
        K = rearrange(k, 'b p1 p2 (x h1) (y h2) (z h3) -> b p1 p2 h1 h2 h3 x y z', h1 = self.h1, h2 = self.h2, h3 = self.h3)
        V = rearrange(v, 'b p1 p2 (x h1) (y h2) (z h3) -> b p1 p2 h1 h2 h3 x y z', h1 = self.h1, h2 = self.h2, h3 = self.h3)


        # Attention calculation
        attn = torch.einsum('b p q d e f x y z,b m n d e f x y z -> b d e f p q m n', Q, K) * self.scale

        softmax_attn = rearrange(attn, 'b h1 h2 h3 p1 p2 q1 q2 -> b (h1 h2 h3) (p1 p2) (q1 q2)')
        softmax_attn = F.softmax(softmax_attn, dim=-1) # rearranged to have the last two modes sum to 1.0
        attention = softmax_attn.view_as(attn)
        
        x = torch.einsum('b d e f p q m n , b m n d e f x y z -> b d e f p q x y z ', attention, V)
        x = rearrange(x, 'b h1 h2 h3 p1 p2 x y z  -> b p1 p2 (x h1) (y h2) (z h3)', h1 = self.h1, h2 = self.h2, h3 = self.h3)

        if self.out_embed:
            x = self.tcl_out(x)

        return x