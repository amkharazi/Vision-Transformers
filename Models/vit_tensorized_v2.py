
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from Models.tensorized_components.patch_embedding import PatchEmbedding
from Models.tensorized_components.encoder_block import Encoder
from Tensorized_Layers.TRL import TRL
from Tensorized_Layers.TCL import TCL


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_size=(16,3,224,224),
                 patch_size=16,
                 num_classes=1000,
                 embed_dim=(16,16,3),
                 num_heads=(2,2,3),
                 num_layers=12,
                 mlp_dim=(16,16,4),
                 dropout=0.1,
                 bias=True,
                 out_embed=True,
                 device='cuda',
                 ignore_modes = (0,1,2),
                 Tensorized_mlp = True
                 ):
        super(VisionTransformer, self).__init__()
        self.device=device
        
        
        self.patch_embedding = PatchEmbedding(input_size=input_size,
                                                patch_size=patch_size,
                                                embed_dim=embed_dim,
                                                bias=bias,
                                                device=device,
                                                ignore_modes= ignore_modes)

        self.pos_embedding = nn.Parameter(
            torch.randn(1,
                        (input_size[2] // patch_size),
                        (input_size[2] // patch_size),
                        embed_dim[0],
                        embed_dim[1],
                        embed_dim[2],
                        device = device
                        ), requires_grad=True)

        self.cls_token = nn.Parameter(
            torch.randn(1,
                        1, 
                        1, 
                        embed_dim[0], 
                        embed_dim[1], 
                        embed_dim[2], 
                        device=device
                        ), requires_grad=True)


        tcl_input_size = (input_size[0], input_size[2]//patch_size + 1, input_size[3]//patch_size,
                                embed_dim[0], embed_dim[1], embed_dim[2]) # patched input image size
        MLP = nn.Sequential(
                TCL(input_size=tcl_input_size, rank=mlp_dim, ignore_modes=ignore_modes, bias=bias, device=device),
                nn.GELU(),
                TCL(input_size=mlp_dim, rank=embed_dim, ignore_modes=ignore_modes, bias=bias, device=device),
                nn.Dropout(dropout),
            )
        norm = nn.LayerNorm(embed_dim)

        self.transformer = nn.ModuleList([
            Encoder(input_size=input_size,
                        patch_size=patch_size,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        bias=bias,out_embed=out_embed,
                        device=device,ignore_modes=ignore_modes,
                        Tensorized_mlp=[MLP, norm]) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # embed layer is the output of final mlp layer in the transformer block
        self.classifier = TRL(input_size=(input_size[0], embed_dim[0], embed_dim[1], embed_dim[2]),
                            output=(num_classes,),
                            rank=(embed_dim[0], embed_dim[1], embed_dim[2], num_classes),
                            ignore_modes=(0,),
                            bias=bias,
                            device=device) # trl rank the same

    def forward(self, x):
        patches = self.patch_embedding(x)

        patches += self.pos_embedding

        ## cls token addon        
        tensor_cls_token = torch.zeros((patches.shape[0],
                                  1,
                                  patches.shape[2],
                                  patches.shape[3],
                                  patches.shape[4],
                                  patches.shape[5])).to(self.device)

        tensor_cls_token[:, 0, 0 ,:,:,:] = self.cls_token

        x = torch.cat([tensor_cls_token, patches ], dim = 1 ).to(self.device)
        
        for transformer_block in self.transformer:
            x = transformer_block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0, 0, :,:,:]
        
        # pass to trl
        # cls_token_final_vec = cls_token_final.flatten().reshape(x.shape[0],-1)
        output = self.classifier(cls_token_final)
        return output
