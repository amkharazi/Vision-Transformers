
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from models.basic_components.patch_embedding import PatchEmbedding
from models.basic_components.encoder_block import Encoder


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_size=(16,3,224,224),
                 patch_size=16,
                 num_classes=1000,
                 embed_dim=3*16*16,
                 num_heads=12,
                 num_layers=12,
                 mlp_dim=1024,
                 dropout=0.1,
                 bias=True,
                 out_embed=True,
                 drop_path =0.1,
                 ):
        super(VisionTransformer, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels=input_size[1],
                                                patch_size=patch_size,
                                                embed_dim=embed_dim,
                                                bias=bias,
                                                )        
                
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad= True)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, ((input_size[2] // patch_size) ** 2) + 1, embed_dim), requires_grad= True)

        self.transformer = nn.ModuleList([
            Encoder(input_size=input_size,
                        patch_size=patch_size,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        dropout=dropout,
                        bias=bias,
                        out_embed=out_embed,
                        drop_path=drop_path,) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier= nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)


    def forward(self, x):
        patches = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        x += self.pos_embedding

        for transformer_block in self.transformer:
            x = transformer_block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        output = self.classifier(cls_token_final)
        return output
