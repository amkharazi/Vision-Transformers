
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, out_embed = True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.h1 = num_heads[0]
        self.h2 = num_heads[1]
        self.h3 = num_heads[2]
        
        self.scale = ((self.embed_dim[0]//self.h1)*
                      (self.embed_dim[1]//self.h2)*
                      (self.embed_dim[2]//self.h3))** -0.5
        
        self.out_embed = out_embed

        # first dim is the dimension of embedded x second dim is the embedded dimension of q/k/v (could be different)
        #  Q
        self.w_e1_q = nn.Parameter(torch.randn(self.embed_dim[0], self.embed_dim[0]), requires_grad=True)
        self.w_e2_q = nn.Parameter(torch.randn(self.embed_dim[1], self.embed_dim[1]), requires_grad=True)
        self.w_e3_q = nn.Parameter(torch.randn(self.embed_dim[2], self.embed_dim[2]), requires_grad=True)

        # K
        self.w_e1_k = nn.Parameter(torch.randn(self.embed_dim[0], self.embed_dim[0]), requires_grad=True)
        self.w_e2_k = nn.Parameter(torch.randn(self.embed_dim[1], self.embed_dim[1]), requires_grad=True)
        self.w_e3_k = nn.Parameter(torch.randn(self.embed_dim[2], self.embed_dim[2]), requires_grad=True)

        # V
        self.w_e1_v = nn.Parameter(torch.randn(self.embed_dim[0], self.embed_dim[0]), requires_grad=True)
        self.w_e2_v = nn.Parameter(torch.randn(self.embed_dim[1], self.embed_dim[1]), requires_grad=True)
        self.w_e3_v = nn.Parameter(torch.randn(self.embed_dim[2], self.embed_dim[2]), requires_grad=True)

        if self.out_embed:
            self.w_e1_out = nn.Parameter(torch.randn(self.embed_dim[0], self.embed_dim[0]), requires_grad=True)
            self.w_e2_out = nn.Parameter(torch.randn(self.embed_dim[1], self.embed_dim[1]), requires_grad=True)
            self.w_e3_out = nn.Parameter(torch.randn(self.embed_dim[2], self.embed_dim[2]), requires_grad=True)

    def forward(self, x):
        q = torch.einsum('b p q h w c , h x , w y , c z -> b p q x y z', (x, self.w_e1_q, self.w_e2_q, self.w_e3_q))
        k = torch.einsum('b p q h w c , h x , w y , c z -> b p q x y z', (x, self.w_e1_k, self.w_e2_k, self.w_e3_k))
        v = torch.einsum('b p q h w c , h x , w y , c z -> b p q x y z', (x, self.w_e1_v, self.w_e2_v, self.w_e3_v))

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
            print(x.shape)
            x = torch.einsum('b p q h w c , h x , w y , c z -> b p q x y z', (x, self.w_e1_out, self.w_e2_out, self.w_e3_out))
        return x