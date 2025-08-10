import sys
sys.path.append(".")

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm

from utils.num_param import param_counts
from utils.flops import try_thop_gflops


class VisionTransformer(nn.Module):
    """
    Wrapper for timm Vision Transformers with flexible freezing.

    Parameters
    ----------
    model_name : str, default "vit_base_patch16_224"
        Any ViT-like model name recognized by timm.create_model.
    num_classes : int, default 1000
        Number of output classes for the classifier head.
    pretrained : bool, default False
        Load pretrained weights if available.
    freeze_encoder : bool, default False
        If True, freeze all encoder params; head remains trainable.
    trainable_blocks : int, default 0
        If >0, unfreezes the last `k` blocks (and norm/head) for finetuning.
    global_pool : Optional[str], default "avg"
        Global pooling mode passed to timm (e.g., "avg", "token", or None/"").
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 1000,
        pretrained: bool = False,
        freeze_encoder: bool = False,
        trainable_blocks: int = 0,
        global_pool: Optional[str] = "avg",
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            global_pool=global_pool if global_pool is not None else "",
        )

        if freeze_encoder:
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            head = self.model.get_classifier()
            if isinstance(head, nn.Module):
                for p in head.parameters():
                    p.requires_grad = True

        if trainable_blocks > 0:
            self._unfreeze_last_blocks(trainable_blocks)

    @property
    def num_features(self) -> int:
        head = getattr(self.model, "get_classifier", None)
        if callable(head):
            clf = self.model.get_classifier()
            if isinstance(clf, nn.Linear):
                return int(clf.in_features)
        return int(getattr(self.model, "num_features", 0))

    def _unfreeze_last_blocks(self, k: int) -> None:
        blocks = getattr(self.model, "blocks", None)
        if isinstance(blocks, (nn.Sequential, list, tuple)) and len(blocks) > 0:
            for p in self.model.parameters():
                p.requires_grad = False
            head = self.model.get_classifier()
            if isinstance(head, nn.Module):
                for p in head.parameters():
                    p.requires_grad = True
            for blk in list(blocks)[-int(k):]:
                for p in blk.parameters():
                    p.requires_grad = True
            for attr in ("norm", "ln2", "fc_norm", "pre_logits"):
                m = getattr(self.model, attr, None)
                if isinstance(m, nn.Module):
                    for p in m.parameters():
                        p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        return self.model(x)


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    warmup: int = 10,
    iters: int = 50,
) -> Tuple[float, float]:
    """
    Benchmark forward-only latency and throughput on random data.

    Returns
    -------
    (latency_ms_per_image, throughput_images_per_second)
    """
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
    """
    Measure average training step time with random inputs/labels (forward+backward+update).
    """
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


def param_breakdown_timm(vit: VisionTransformer) -> Tuple[int, int, int, int, int, int]:
    """
    Return (total, trainable, patch_embed, blocks, head, other) parameter counts.
    Works for timm ViT-like models that expose .patch_embed, .blocks, .norm, .head.
    """
    total_params, trainable_params = param_counts(vit)

    m = vit.model
    pe = param_counts(getattr(m, "patch_embed", nn.Identity()))[0] if hasattr(m, "patch_embed") else 0
    blocks = getattr(m, "blocks", None)
    blk = sum(param_counts(b)[0] for b in (list(blocks) if isinstance(blocks, (list, tuple)) else (blocks or []))) if blocks is not None else 0
    head_mod = m.get_classifier() if hasattr(m, "get_classifier") else getattr(m, "head", None)
    head = param_counts(head_mod)[0] if isinstance(head_mod, nn.Module) else 0

    counted = pe + blk + head
    other = total_params - counted
    return total_params, trainable_params, pe, blk, head, other


def sanity_and_benchmark() -> None:
    """
    Build a timm ViT, verify shapes, print params (+breakdown),
    THOP GFLOPs (if available), inference latency/throughput,
    and synthetic train step time.
    """
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    model_name = "vit_base_patch16_224"
    num_classes = 10
    B, C, H, W = 16, 3, 224, 224
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit = VisionTransformer(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_encoder=True,
        trainable_blocks=2,
        global_pool="avg",
    ).to(device)

    x = torch.randn(B, C, H, W, device=device)
    y = vit(x)
    assert y.shape == (B, num_classes), f"Expected {(B, num_classes)}, got {y.shape}"

    total, trainable, pe, blk, head, other = param_breakdown_timm(vit)
    print(f"Params: total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M")
    print(f"  - patch_embed={pe/1e6:.2f}M | blocks={blk/1e6:.2f}M | head={head/1e6:.2f}M | other={other/1e6:.2f}M")

    gflops = try_thop_gflops(vit, (B, C, H, W))
    print(f"THOP GFLOPs @ {H}x{W}: {'n/a' if gflops is None else f'{float(gflops):.2f}'}")

    lat_ms, tput = benchmark_inference(vit, (B, C, H, W), warmup=10, iters=50)
    print(f"Inference: latency={lat_ms:.2f} ms/img, throughput={tput:.1f} img/s")

    avg_step_ms = train_one_epoch_synthetic(vit, (B, C, H, W), num_classes, steps=20, lr=1e-3)
    print(f"Synthetic train (1 epoch, {20} steps): avg step={avg_step_ms:.2f} ms")

    print("VisionTransformer benchmark sanity check passed.")


if __name__ == "__main__":
    sanity_and_benchmark()
