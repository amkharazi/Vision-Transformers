import sys
sys.path.append(".")

import time
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn

from models.tensorized_components.patch_embedding import PatchEmbedding
from models.tensorized_components.encoder_block import Encoder
from tensorized_layers.TP import TP
from utils.num_param import count_parameters, param_counts
from utils.flops import try_flops_gflops


class VisionTransformer(nn.Module):
    """
    Tensorized Vision Transformer with single-column CLS.

    Tokens: (B, P_h+1, P_w, d1, d2, d3). The CLS row is zero everywhere except column 0,
    which holds a learnable token. Positional embedding shape matches your original
    (P_h+1, P_h), assuming square inputs.

    Parameters
    ----------
    input_size : Sequence[int], default (16,3,224,224)
    patch_size : int, default 16
    num_classes : int, default 1000
    embed_dim : Tuple[int,int,int], default (3,16,16)
    num_heads : Tuple[int,int,int], default (2,2,3)
    num_layers : int, default 12
    mlp_dim : Tuple[int,int,int], default (3,16,16)
    dropout : float, default 0.1
    bias : bool, default True
    out_embed : bool, default True
    drop_path : float, default 0.1
    ignore_modes : Iterable[int], default (0,1,2)
    tensor_method_mlp : Tuple[str,str], default ('tle','tle')
    tensor_method : {'tle','tdle','tp'}, default 'tle'
    tdle_level : int, default 3
    reduce_level : Tuple[int,int,int], default (0,0,0)
    """

    def __init__(
        self,
        input_size: Sequence[int] = (16, 3, 224, 224),
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: Tuple[int, int, int] = (3, 16, 16),
        num_heads: Tuple[int, int, int] = (2, 2, 3),
        num_layers: int = 12,
        mlp_dim: Tuple[int, int, int] = (3, 16, 16),
        dropout: float = 0.1,
        bias: bool = True,
        out_embed: bool = True,
        drop_path: float = 0.1,
        ignore_modes: Iterable[int] = (0, 1, 2),
        tensor_method_mlp: Tuple[str, str] = ("tle", "tle"),
        tensor_method: str = "tle",
        tdle_level: int = 3,
        reduce_level: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()

        if not (isinstance(embed_dim, tuple) and len(embed_dim) == 3):
            raise TypeError("embed_dim must be a 3-tuple")
        if not (isinstance(mlp_dim, tuple) and len(mlp_dim) == 3):
            raise TypeError("mlp_dim must be a 3-tuple")
        if not (isinstance(num_heads, tuple) and len(num_heads) == 3):
            raise TypeError("num_heads must be a 3-tuple")
        if not (isinstance(input_size, Sequence) and len(input_size) == 4):
            raise TypeError("input_size must be (B, C, H, W)")

        B, C, H, W = map(int, input_size)
        if H % patch_size or W % patch_size:
            raise ValueError(f"H and W must be divisible by patch_size={patch_size}")

        P_h, P_w = H // patch_size, W // patch_size
        self.input_size = (B, C, H, W)
        self.patch_size = int(patch_size)
        self.embed_dim = embed_dim
        self.num_classes = int(num_classes)

        self.patch_embedding = PatchEmbedding(
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            bias=bias,
            ignore_modes=ignore_modes,
            tensor_method=tensor_method,
            tdle_level=tdle_level,
            reduce_level=reduce_level,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, *embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, P_h + 1, P_h, *embed_dim))  # kept as you had

        self.transformer = nn.ModuleList(
            [
                Encoder(
                    input_size=input_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    bias=bias,
                    out_embed=out_embed,
                    drop_path=drop_path,
                    ignore_modes=ignore_modes,
                    tensor_method_mlp=tensor_method_mlp,
                    tensor_method=tensor_method,
                    tdle_level=tdle_level,
                    reduce_level=reduce_level,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = TP(
            input_size=(B, *embed_dim),
            output_size=(num_classes,),
            rank=tuple(x - y for x, y in zip(embed_dim + (num_classes,), reduce_level + (0,))),
            ignore_modes=(0,),
            bias=bias,
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        returns: (B, num_classes)
        """
        patches = self.patch_embedding(x)  # (B, P_h, P_w, d1,d2,d3)

        tensor_cls = torch.zeros(
            (patches.shape[0], 1, patches.shape[2], *self.embed_dim),
            dtype=patches.dtype,
            device=patches.device,
        )
        tensor_cls[:, 0, 0, :, :, :] = self.cls_token  # only column 0 holds the cls token

        x_tok = torch.cat([tensor_cls, patches], dim=1)  # (B, P_h+1, P_w, d1,d2,d3)
        x_tok = x_tok + self.pos_embedding.to(x_tok.device)  # pos embedding kept as defined

        for blk in self.transformer:
            x_tok = blk(x_tok)

        x_tok = self.norm(x_tok)
        cls_token_final = x_tok[:, 0, 0]  # (B, d1,d2,d3)
        return self.classifier(cls_token_final)


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    warmup: int = 10,
    iters: int = 50,
) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    x = torch.randn(*input_size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    for _ in range(warmup):
        _ = model(x)
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    sync()
    total_s = time.perf_counter() - t0

    latency_ms = (total_s / iters) * 1000.0 / input_size[0]
    throughput = (iters * input_size[0]) / total_s
    return latency_ms, throughput


def train_one_epoch_synthetic(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    num_classes: int,
    steps: int = 20,
    lr: float = 1e-3,
) -> float:
    device = next(model.parameters()).device
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    times = []
    for _ in range(steps):
        x = torch.randn(*input_size, device=device)
        y = torch.randint(0, num_classes, (input_size[0],), device=device)

        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(sum(times) / len(times))


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    B, C, H, W = 2, 3, 224, 224
    ps = 16
    num_classes = 10
    embed_dim = (3, 16, 16)
    heads = (1, 2, 2)
    mlp_dim = (4, 16, 16)
    num_layers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit = VisionTransformer(
        input_size=(B, C, H, W),
        patch_size=ps,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        tensor_method="tle",
        tensor_method_mlp=("tle", "tle"),
    ).to(device)

    x = torch.randn(B, C, H, W, device=device)
    y = vit(x)
    assert y.shape == (B, num_classes), f"Expected {(B, num_classes)}, got {y.shape}"

    total_params, trainable_params = param_counts(vit)
    print(f"Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")

    gflops = try_flops_gflops(vit, (B, C, H, W))
    if gflops is None:
        print("GFLOPs: n/a")
    elif isinstance(gflops, (tuple, list)):
        print(f"GFLOPs @ {H}x{W}: {float(gflops[0]):.2f}")
    else:
        print(f"GFLOPs @ {H}x{W}: {float(gflops):.2f}")

    lat_ms, tput = benchmark_inference(vit, (B, C, H, W), warmup=5, iters=20)
    print(f"Inference: latency={lat_ms:.2f} ms/img, throughput={tput:.1f} img/s")

    avg_step_ms = train_one_epoch_synthetic(vit, (B, C, H, W), num_classes, steps=10, lr=1e-3)
    print(f"Synthetic train: avg step={avg_step_ms:.2f} ms")

    print("Tensorized VisionTransformer sanity + benchmark passed.")
