
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from models.tensorized_components.patch_embedding import PatchEmbedding
from models.tensorized_components.encoder_block import Encoder
from tensorized_layers.TP import TP

class VisionTransformer(nn.Module):
    def __init__(self,
                input_size=(16,3,224,224),
                patch_size=16,
                num_classes=1000,
                embed_dim=(3,16,16),
                num_heads=(2,2,3),
                num_layers=12,
                mlp_dim=(3,16,16),
                dropout=0.1,
                bias=True,
                out_embed=True,
                drop_path =0.1,
                ignore_modes = (0,1,2),
                tensor_method_mlp= ('tle','tle'),
                tensor_method='tle',
                tdle_level = 3,
                reduce_level = (0,0,0),
                ):
        super(VisionTransformer, self).__init__()        
        
        self.patch_embedding = PatchEmbedding(input_size=input_size,
                                                patch_size=patch_size,
                                                embed_dim=embed_dim,
                                                bias=bias,
                                                ignore_modes= ignore_modes, 
                                                tensor_method=tensor_method,
                                                tdle_level = tdle_level,
                                                reduce_level = reduce_level)

        self.cls_token = nn.Parameter(
            torch.randn(1,
                        1, 
                        1, 
                        embed_dim[0], 
                        embed_dim[1], 
                        embed_dim[2], 
                        ), requires_grad=True)
        
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1,
                        (input_size[2] // patch_size) + 1,
                        (input_size[2] // patch_size),
                        embed_dim[0],
                        embed_dim[1],
                        embed_dim[2],
                        ), requires_grad=True)
        
        self.transformer = nn.ModuleList([
            Encoder(input_size=input_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    bias = bias,
                    out_embed = out_embed,
                    drop_path =drop_path,
                    ignore_modes=ignore_modes,
                    tensor_method_mlp= tensor_method_mlp,
                    tensor_method=tensor_method,
                    tdle_level = tdle_level,
                    reduce_level = reduce_level,) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = TP(
            input_size=(input_size[0], *embed_dim),
            output_size=(num_classes,),
            rank = tuple(x - y for x, y in zip(embed_dim + (num_classes,), reduce_level + (0,))),
            ignore_modes=(0,),
            bias=bias,
            )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        patches = self.patch_embedding(x)

        tensor_cls_token = torch.zeros((patches.shape[0],
                                  1,
                                  patches.shape[2],
                                  patches.shape[3],
                                  patches.shape[4],
                                  patches.shape[5]))

        tensor_cls_token[:, 0, 0 ,:,:,:] = self.cls_token

        x = torch.cat([tensor_cls_token, patches ], dim = 1 )

        x += self.pos_embedding
        
        for transformer_block in self.transformer:
            x = transformer_block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0, 0, :,:,:]
        
        output = self.classifier(cls_token_final)
        return output
