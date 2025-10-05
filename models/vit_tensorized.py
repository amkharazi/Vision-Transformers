import sys

sys.path.append("..")
import torch
import torch.nn as nn
from models.tensorized_components.patch_embedding import PatchEmbedding
from models.tensorized_components.encoder_block import Encoder as tensorized_encoder
from models.basic_components.encoder_block import Encoder as basic_encoder
from tensorized_layers.TP import TP


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_size=(16, 3, 224, 224),
        patch_size=16,
        num_classes=1000,
        embed_dim=(16, 16, 3),
        num_heads=(2, 2, 3),
        num_layers=12,
        mlp_dim=(16, 16, 4),
        dropout=0.1,
        bias=True,
        out_embed=True,
        device="cuda",
        ignore_modes=(0, 1, 2),
        Tensorized_mlp=True,
        tensor_type=("tle", "tle"),
        tdle_level=3,
        num_tensorized="full",
    ):
        super(VisionTransformer, self).__init__()
        self.device = device

        self.num_layers = num_layers
        if num_tensorized != "full":
            self.num_tensorized = int(num_tensorized)
        else:
            self.num_tensorized = num_layers

        self.patch_embedding = PatchEmbedding(
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            bias=bias,
            device=device,
            ignore_modes=ignore_modes,
            tensor_type=tensor_type[0],
            tdle_level=tdle_level,
        )

        self.cls_token = nn.Parameter(
            torch.randn(
                1, 1, 1, embed_dim[0], embed_dim[1], embed_dim[2], device=device
            ),
            requires_grad=True,
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                (input_size[2] // patch_size) + 1,
                (input_size[2] // patch_size),
                embed_dim[0],
                embed_dim[1],
                embed_dim[2],
                device=device,
            ),
            requires_grad=True,
        )

        self.transformer_tensorized = nn.ModuleList(
            [
                tensorized_encoder(
                    input_size=input_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    bias=bias,
                    out_embed=out_embed,
                    device=device,
                    ignore_modes=ignore_modes,
                    Tensorized_mlp=Tensorized_mlp,
                    tensor_type=tensor_type,
                    tdle_level=tdle_level,
                )
                for _ in range(self.num_tensorized)
            ]
        )

        self.transformer_basic = nn.ModuleList(
            [
                basic_encoder(
                    input_size=input_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim[0] * embed_dim[1] * embed_dim[2],
                    num_heads=num_heads[0] * num_heads[1] * num_heads[2],
                    mlp_dim=mlp_dim[0] * mlp_dim[1] * mlp_dim[2],
                    dropout=dropout,
                    bias=bias,
                    out_embed=out_embed,
                    device=device,
                    ignore_modes=None,
                    Tensorized_mlp=False,
                )
                for _ in range(self.num_layers - self.num_tensorized)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.norm_base = nn.LayerNorm(embed_dim[0] * embed_dim[1] * embed_dim[2])
        self.classifier = TP(
            input_size=(input_size[0], embed_dim[0], embed_dim[1], embed_dim[2]),
            output=(num_classes,),
            rank=(embed_dim[0], embed_dim[1], embed_dim[2], num_classes),
            ignore_modes=(0,),
            bias=bias,
            device=device,
        )
        self.classifier_mlp = nn.Linear(
            embed_dim[0] * embed_dim[1] * embed_dim[2], num_classes, bias=bias
        )

    def forward(self, x):
        patches = self.patch_embedding(x)

        tensor_cls_token = torch.zeros(
            (
                patches.shape[0],
                1,
                patches.shape[2],
                patches.shape[3],
                patches.shape[4],
                patches.shape[5],
            )
        ).to(self.device)

        tensor_cls_token[:, 0, 0, :, :, :] = self.cls_token

        x = torch.cat([tensor_cls_token, patches], dim=1).to(self.device)

        x += self.pos_embedding

        for transformer_block in self.transformer_tensorized:
            x = transformer_block(x)
        if self.num_tensorized != "full":
            x = torch.cat(
                (
                    x[:, 0, 0, :, :, :].reshape(patches.shape[0], -1).unsqueeze(1),
                    x[:, 1:, :, :, :, :].reshape(
                        patches.shape[0],
                        patches.shape[1] * patches.shape[2],
                        patches.shape[3] * patches.shape[4] * patches.shape[5],
                    ),
                ),
                dim=1,
            )
            for transformer_block in self.transformer_basic:
                x = transformer_block(x)
        if self.num_tensorized != "full":
            x = self.norm_base(x)
            cls_token_final = x[:, 0, :]
            output = self.classifier_mlp(cls_token_final)
        else:
            x = self.norm(x)
            cls_token_final = x[:, 0, 0, :, :, :]
            output = self.classifier(cls_token_final)
        return output
