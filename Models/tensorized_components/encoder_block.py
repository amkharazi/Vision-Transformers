
import sys
sys.path.append('..')
import torch.nn as nn
from Models.tensorized_components.multihead_attention import MultiHeadAttention as MHA
from einops import rearrange
from Tensorized_Layers.TCL import TCL, TCL_extended
from Tensorized_Layers.TRL import TRL

class Encoder(nn.Module):
    def __init__(self,input_size, patch_size, embed_dim, num_heads, mlp_dim, dropout=0.5, bias = True, out_embed = True, device = 'cuda', ignore_modes=(0,1,2), Tensorized_mlp = True, tcl_type='normal', tcl_r = 3):
        super(Encoder, self).__init__()
        self.tensorized_mlp = Tensorized_mlp
        self.tcl_input_size = (input_size[0], input_size[2]//patch_size + 1, input_size[3]//patch_size,
                                embed_dim[0], embed_dim[1], embed_dim[2]) # patched input image size
        self.trl_input_size = (input_size[0], input_size[2]//patch_size + 1, input_size[3]//patch_size,
                                mlp_dim[0], mlp_dim[1], mlp_dim[2])
        self.trl_ranks = tuple([i for i in mlp_dim+embed_dim])  # same rank for trl
        # print(self.trl_ranks)
        # print(mlp_dim+embed_dim)


        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = MHA(
                            input_size=input_size,
                            patch_size=patch_size,  
                            embed_dim= embed_dim,
                            num_heads=num_heads,
                            bias=bias,
                            out_embed= out_embed,
                            device=device,
                            ignore_modes=ignore_modes,  
                            tcl_type=tcl_type, 
                            tcl_r = tcl_r
                             )
        
        if self.tensorized_mlp == False:
            feature_dim = embed_dim[0]*embed_dim[1]*embed_dim[2]
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, feature_dim),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(feature_dim)
        elif self.tensorized_mlp == True:
            if tcl_type=='normal':
                self.mlp = nn.Sequential(
                    TCL(input_size=self.tcl_input_size, rank=mlp_dim, ignore_modes=ignore_modes, bias=bias, device=device),
                    nn.GELU(),
                    TRL(input_size=self.trl_input_size, output=embed_dim, rank=self.trl_ranks, ignore_modes=ignore_modes, bias=bias,device=device),
                    nn.Dropout(dropout),
                )
            else:
                self.mlp = nn.Sequential(
                    TCL_extended(input_size=self.tcl_input_size, rank=mlp_dim, ignore_modes=ignore_modes, bias=bias, device=device, r=tcl_r),
                    nn.GELU(),
                    TRL(input_size=self.trl_input_size, output=embed_dim, rank=self.trl_ranks, ignore_modes=ignore_modes, bias=bias,device=device),
                    nn.Dropout(dropout),
                )
        else:
            self.mlp = self.tensorized_mlp[0]
            self.norm2 = self.tensorized_mlp[1]
        

    def forward(self, x):
        x_res = x
        x = self.norm1(x + self.dropout(self.attention(x)))
        if self.tensorized_mlp==False:
            shapes = x.shape
            x = rearrange(x, 'b p1 p2 h w c -> b (p1 p2) (h w c)')
            x = self.norm2(x + self.dropout(self.mlp(x)))
            x = rearrange(x, 'b (p1 p2) (h w c) -> b p1 p2 h w c', p1 = shapes[1], p2 = shapes[2], h = shapes[3], w = shapes[4], c = shapes[5])
        else:
            x = self.norm2(x + self.dropout(self.mlp(x)))
        return x + x_res