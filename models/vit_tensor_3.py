# Tensor 1 : MLP layer only TLE then TP.
# scale 0.25
# Vectorized tokens order

import sys
sys.path.append(".")

from einops import rearrange
import torch
import torch.nn as nn
from utils.num_param import count_parameters
from tensorized_layers.TLE import TLE
from tensorized_layers.TP import TP

class DropPath(nn.Module):
    def __init__(self, drop_prob: 0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * rand

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, embed_dim, batch_size = 32, ignore_modes = (0,1), bias = True):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.proj = TLE(
            input_size=(batch_size, self.num_patches, in_chans, patch_size, patch_size),
            ranks= embed_dim,
            bias=bias,
            ignore_modes=ignore_modes)

    def forward(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        print(f'patch embedding (1): {x.shape}')
        x = self.proj(x)
        print(f'patch embedding (2): {x.shape}')
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=(3,2,2), bias=True, batch_size = 32):
        super().__init__()
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.head_dim_1 = dim[0] // num_heads[0]
        self.head_dim_2 = dim[1] // num_heads[1]
        self.head_dim_3 = dim[2] // num_heads[2]
        # self.scale = (head_dim_1 * head_dim_2 * head_dim_3) ** 0.5
        self.scale = (self.head_dim_1 * self.head_dim_2 * self.head_dim_3) ** 0.25
        
        self.q_proj = TLE(
            input_size=(batch_size, self.num_patches, dim[0], dim[1], dim[2]),
            ranks= (dim[0], dim[1], dim[2]),
            bias=bias,
            ignore_modes=(0,1))
        
        self.k_proj = TLE(
            input_size=(batch_size, self.num_patches, dim[0], dim[1], dim[2]),
            ranks= (dim[0], dim[1], dim[2]),
            bias=bias,
            ignore_modes=(0,1))
        
        self.v_proj = TLE(
            input_size=(batch_size, self.num_patches, dim[0], dim[1], dim[2]),
            ranks= (dim[0], dim[1], dim[2]),
            bias=bias,
            ignore_modes=(0,1))
        
        self.proj = TLE(
            input_size=(batch_size, self.num_patches, dim[0], dim[1], dim[2]),
            ranks= (dim[0], dim[1], dim[2]),
            bias=bias,
            ignore_modes=(0,1))

    def forward(self, x):
        B, N, C, H, W = x.shape
        print(f'Attention: {x.shape}')
        q = self.q_proj(x)
        print(f'Q (1): {q.shape}')
        q = rearrange(q,'b n (h0 d0) (h1 d1) (h2 d2) -> b h0 h1 h2 n d0 d1 d2',h0=self.num_heads[0],h1=self.num_heads[1],h2=self.num_heads[2])
        print(f'Q (2): {q.shape}')
        k = self.k_proj(x)
        print(f'K (1): {k.shape}')
        k = rearrange(k,'b n (h0 d0) (h1 d1) (h2 d2) -> b h0 h1 h2 n d0 d1 d2',h0=self.num_heads[0],h1=self.num_heads[1],h2=self.num_heads[2])
        print(f'K (2): {k.shape}')
        v = self.v_proj(x)
        print(f'V (1): {v.shape}')
        v = rearrange(v,'b n (h0 d0) (h1 d1) (h2 d2) -> b h0 h1 h2 n d0 d1 d2',h0=self.num_heads[0],h1=self.num_heads[1],h2=self.num_heads[2])
        print(f'V (2): {v.shape}')
        
        attn = torch.einsum(
            'b a c d i x y z, b a c d I x y z -> b a c d i I',
            q, k
        ) * self.scale
        print(f'Attention Score (1) {attn.shape}')
        attn = attn.softmax(dim=-1)
        x = torch.einsum(
            'b a c d i I, b a c d I x y z -> b a c d i x y z',
            attn, v
        )
        print(f'QKV (1) {x.shape}')
        x = rearrange(x, 'b a c d i x y z -> b i (a x) (c y) (d z)')
        print(f'QKV (2) {x.shape}')
        x = self.proj(x)
        print(f'Out {x.shape}')
        return x

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, num_patches, drop=0., batch_size = 32, bias=True):
        super().__init__()
        self.num_patches=num_patches
        self.layer1 = TLE(input_size=(batch_size, self.num_patches, dim[0], dim[1], dim[2]),
                          ranks=(hidden_dim[0], hidden_dim[1], hidden_dim[2]), 
                          bias=bias, 
                          ignore_modes=(0,1))
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.layer2 = TP(input_size=(batch_size, self.num_patches, hidden_dim[0], hidden_dim[1], hidden_dim[2]), 
                          output_size=(dim[0], dim[1], dim[2]),
                          ranks=(hidden_dim[0], hidden_dim[1], hidden_dim[2], dim[0], dim[1], dim[2]), 
                          bias=bias, 
                          ignore_modes=(0,1))
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        print(f'MLP (1) {x.shape}')
        x = self.layer1(x)
        print(f'MLP (2) {x.shape}')
        x = self.act(x)
        x = self.drop1(x)
        x = self.layer2(x)
        print(f'MLP (3) {x.shape}')
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_patches, num_heads, mlp_dim, bias=True, drop=0.0, drop_path=0.0, batch_size=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_patches=num_patches, num_heads=num_heads, bias=bias, batch_size=batch_size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, hidden_dim=mlp_dim, num_patches=num_patches, drop=drop, batch_size=batch_size, bias=bias)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        batch_size = 32,
        image_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=(3,16,16),
        depth=12,
        num_heads=(3, 2, 2),
        mlp_dim=(3,32,32),
        bias=True,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, batch_size=batch_size, ignore_modes=(0,1), bias=bias)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[0], embed_dim[1], embed_dim[2]))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim[0], embed_dim[1], embed_dim[2]))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                bias=bias,
                drop=drop,
                drop_path=drop_path,
                num_patches=(num_patches + 1), 
                batch_size=batch_size
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = TP(input_size=(batch_size, embed_dim[0], embed_dim[1], embed_dim[2]),
                       output_size=(num_classes, ), ranks=(embed_dim[0], embed_dim[1], embed_dim[2], num_classes),
                       bias=bias,
                       ignore_modes=(0,))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        print(f'start {x.shape}')
        x = self.patch_embed(x)
        print(f'after patch embedding {x.shape}')
        B, N, C ,H, W = x.shape
        print(f'cls token before expansion {self.cls_token.shape}')
        cls_token = self.cls_token.expand(B, -1, -1, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        print(f'cls token integrated : {x.shape}')
        x = x + self.pos_embed
        print(f'after pos embedding : {x.shape}')
        for blk in self.blocks:
            print(f'before block {x.shape}')
            x = blk(x)
            print(f'after block {x.shape}')
        x = self.norm(x)
        print(f'after norm {x.shape}')
        x = x[:, 0, :, :, :]
        print(f'cls extracted {x.shape}')
        x = self.head(x)
        print(f'after classification head {x.shape}')
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionTransformer(
        image_size=224,
        patch_size=16,  
        in_chans=3,
        num_classes=200,
        embed_dim=(3,16,16),
        depth=12,
        num_heads=(3,2,2),
        mlp_dim=(3, 32,32),
        bias=True,
        drop=0.0,
        drop_path=0.0,
    ).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        _ = model(x)
    print("Our tensor vit params:", count_parameters(model))
