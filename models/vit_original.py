import sys
sys.path.append(".")

import time
from typing import Tuple, Optional

import torch
import torch.nn as nn

from models.basic_components.patch_embedding import PatchEmbedding
from models.basic_components.encoder_block import Encoder
from utils.num_param import param_counts
from utils.flops import (
    conv_patch_embed_flops,
    linear_flops,
    encoder_block_flops,
    layernorm_flops,
    to_gflops,
)


class VisionTransformer(nn.Module):
    """
    Vision Transformer using custom PatchEmbedding and Encoder blocks.

    Parameters
    ----------
    input_size : tuple[int, int, int, int], default (16, 3, 224, 224)
        (B, C, H, W) used for positional embedding sizing.
    patch_size : int, default 16
        Square patch size (H and W must be divisible by this).
    num_classes : int, default 1000
        Number of output classes.
    embed_dim : int, default 3*16*16
        Token embedding dimension D.
    num_heads : int, default 12
        Attention heads in each encoder block.
    num_layers : int, default 12
        Number of encoder blocks.
    mlp_dim : int, default 1024
        Hidden dimension for the MLP inside each encoder.
    dropout : float, default 0.1
    bias : bool, default True
    out_embed : bool, default True
        Whether attention uses an output projection.
    drop_path : float, default 0.1
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int, int] = (16, 3, 224, 224),
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int = 3 * 16 * 16,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        bias: bool = True,
        out_embed: bool = True,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()

        B, C, H, W = input_size
        if H % patch_size or W % patch_size:
            raise ValueError(f"H and W must be divisible by patch_size={patch_size}")
        num_patches = (H // patch_size) * (W // patch_size)

        self.patch_embedding = PatchEmbedding(
            in_channels=C,
            patch_size=patch_size,
            embed_dim=embed_dim,
            bias=bias,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.transformer = nn.ModuleList([
            Encoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                bias=bias,
                out_embed=out_embed,
                drop_path=drop_path,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self._cfg = {
            "input_size": input_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_dim": mlp_dim,
            "bias": bias,
            "out_embed": out_embed,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, C, H, W)

        Returns
        -------
        torch.Tensor
            (B, num_classes)
        """
        patches = self.patch_embedding(x)                  # (B, N, D)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)       # (B, N+1, D)
        x = x + self.pos_embedding[:, : x.size(1)]

        for block in self.transformer:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
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


def analytic_vit_flops(model: VisionTransformer, input_size: Tuple[int, int, int, int]) -> Tuple[float, dict]:
    """
    Analytic GFLOPs for the whole ViT: patch embed + blocks + final norm + head + pos add.
    Returns (gflops_total, breakdown_dict).
    """
    B, C, H, W = input_size
    ps = model._cfg["patch_size"]
    D = model._cfg["embed_dim"]
    L = model._cfg["num_layers"]
    Hh = model._cfg["num_heads"]
    M = model._cfg["mlp_dim"]
    bias = bool(model._cfg["bias"])
    out_proj = bool(model._cfg["out_embed"])

    N = (H // ps) * (W // ps)
    seq = N + 1

    fl_patch = conv_patch_embed_flops(B, C, H, W, ps, D, include_bias=bias)
    fl_pos = B * seq * D

    fl_blocks = 0
    per_block = []
    for _ in range(L):
        parts = encoder_block_flops(
            batch=B, tokens=seq, dim=D, num_heads=Hh, mlp_dim=M,
            out_proj=out_proj, include_bias=bias, include_layernorm=True,
            include_residual=True, drop_path_rate=0.0
        )
        per_block.append(parts["total"])
        fl_blocks += parts["total"]

    fl_norm = layernorm_flops(B, seq, D)
    fl_head = linear_flops(B, D, model.classifier.out_features, include_bias=True)

    total = fl_patch + fl_pos + fl_blocks + fl_norm + fl_head
    breakdown = {
        "patch_embed": fl_patch,
        "pos_add": fl_pos,
        "blocks_total": fl_blocks,
        "block_avg": int(fl_blocks // L) if L > 0 else 0,
        "final_norm": fl_norm,
        "classifier": fl_head,
    }
    return float(total) / 1e9, breakdown


def try_thop_gflops(model: nn.Module, input_size: Tuple[int, int, int, int]) -> Optional[float]:
    """
    Optional GFLOPs via thop.profile; returns None if thop isn't available.
    """
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        dummy = torch.empty(*input_size, device=device)
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(flops) / 1e9
    except Exception:
        return None
    finally:
        if was_training:
            model.train()


def sanity_and_benchmark() -> None:
    """
    Build ViT, verify shapes, print params, analytic & optional THOP GFLOPs,
    inference latency/throughput, and synthetic train step time.
    """
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    B, C, H, W = 16, 3, 32, 32
    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit = VisionTransformer(
        input_size=(B, C, H, W),
        patch_size=4,
        num_classes=num_classes,
        embed_dim=64,
        num_heads=8,
        num_layers=6,
        mlp_dim=128,
        dropout=0.1,
        bias=True,
        out_embed=True,
        drop_path=0.1,
    ).to(device)

    x = torch.randn(B, C, H, W, device=device)
    y = vit(x)
    assert y.shape == (B, num_classes)

    total_params, trainable_params = param_counts(vit)
    print(f"Params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")

    gflops_analytic, br = analytic_vit_flops(vit, (B, C, H, W))
    print(f"Analytic GFLOPs @ {H}x{W}: {gflops_analytic:.2f}")
    print(f"  - patch_embed: {to_gflops(br['patch_embed']):.2f} GFLOPs")
    print(f"  - pos_add:     {to_gflops(br['pos_add']):.2f} GFLOPs")
    print(f"  - blocks:      {to_gflops(br['blocks_total']):.2f} GFLOPs (avg {to_gflops(br['block_avg']):.2f})")
    print(f"  - final_norm:  {to_gflops(br['final_norm']):.2f} GFLOPs")
    print(f"  - classifier:  {to_gflops(br['classifier']):.2f} GFLOPs")

    gflops_thop = try_thop_gflops(vit, (B, C, H, W))
    if gflops_thop is None:
        print("THOP GFLOPs: n/a")
    else:
        print(f"THOP GFLOPs @ {H}x{W}: {float(gflops_thop):.2f}")

    lat_ms, tput = benchmark_inference(vit, (B, C, H, W), warmup=5, iters=20)
    print(f"Inference: latency={lat_ms:.2f} ms/img, throughput={tput:.1f} img/s")

    avg_step_ms = train_one_epoch_synthetic(vit, (B, C, H, W), num_classes, steps=20, lr=1e-3)
    print(f"Synthetic train: avg step={avg_step_ms:.2f} ms")

    print("VisionTransformer sanity + benchmark passed.")


if __name__ == "__main__":
    sanity_and_benchmark()
