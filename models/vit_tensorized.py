import sys
sys.path.append(".")

import time
import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.tensorized_components.patch_embedding import PatchEmbedding
from models.tensorized_components.encoder_block import Encoder
from tensorized_layers.TP import TP
from utils.num_param import param_counts
from utils.flops import (
    estimate_tp_flops,
    tle_input_projector_flops,
    bias_add_flops,
    layernorm_flops,
    residual_add_flops,
    droppath_flops,
    to_gflops,
    try_thop_gflops,
)

class VisionTransformer(nn.Module):
    """
    Tensorized Vision Transformer with single-column CLS.

    Tokens: (B, P_h+1, P_w, d1, d2, d3). The CLS row is zero everywhere except column 0,
    which holds a learnable token. Positional embedding shape is (1, P_h+1, P_w, d1, d2, d3).

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
    rank_patch : Optional[Sequence[int]], default None
        Rank used only when patch tensor_method == 'tp'. If None, defaults to (*in_modes, *out_modes).
    rank_attn : Optional[Sequence[int]], default None
        Rank used only when attention tensor_method == 'tp'. If None, defaults to (*in_modes, *out_modes).
    rank_mlp1 : Optional[Sequence[int]], default None
        Rank used only when MLP-1 tensor_method == 'tp'. If None, defaults to (*embed_dim, *mlp_dim).
    rank_mlp2 : Optional[Sequence[int]], default None
        Rank used only when MLP-2 tensor_method == 'tp'. If None, defaults to (*mlp_dim, *embed_dim).
    rank_classifier : Optional[Sequence[int]], default None
        Rank used only for TP classifier mapping embed_dim → (num_classes,). If None, defaults to (*embed_dim, num_classes).
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
        rank_patch: Optional[Sequence[int]] = None,
        rank_attn: Optional[Sequence[int]] = None,
        rank_mlp1: Optional[Sequence[int]] = None,
        rank_mlp2: Optional[Sequence[int]] = None,
        rank_classifier: Optional[Sequence[int]] = None,
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

        def resolve_tp_rank(in_modes: Tuple[int, ...], out_modes: Tuple[int, ...], r: Optional[Sequence[int]]) -> Tuple[int, ...]:
            if r is not None:
                return tuple(int(x) for x in r)
            return tuple(int(m) for m in (*in_modes, *out_modes))

        self.patch_embedding = PatchEmbedding(
            input_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            bias=bias,
            ignore_modes=ignore_modes,
            tensor_method=tensor_method,
            tdle_level=tdle_level,
            rank=rank_patch,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, *embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, P_h + 1, P_w, *embed_dim))

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
                    rank_attn=rank_attn,
                    rank_mlp1=rank_mlp1,
                    rank_mlp2=rank_mlp2,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        clf_rank = resolve_tp_rank(embed_dim, (num_classes,), rank_classifier)
        if len(clf_rank) != len(embed_dim) + 1 or any(int(v) <= 0 for v in clf_rank):
            raise ValueError(f"rank_classifier must be length {len(embed_dim)+1} with positive entries, got {clf_rank}")
        self.classifier = TP(
            input_size=(B, *embed_dim),
            output_size=(num_classes,),
            rank=clf_rank,
            ignore_modes=(0,),
            bias=bias,
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

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
        patches = self.patch_embedding(x)
        tensor_cls = torch.zeros(
            (patches.shape[0], 1, patches.shape[2], *self.embed_dim),
            dtype=patches.dtype,
            device=patches.device,
        )
        tensor_cls[:, 0, 0, :, :, :] = self.cls_token
        x_tok = torch.cat([tensor_cls, patches], dim=1)
        x_tok = x_tok + self.pos_embedding.to(x_tok.device)
        for blk in self.transformer:
            x_tok = blk(x_tok)
        x_tok = self.norm(x_tok)
        cls_token_final = x_tok[:, 0, 0]
        return self.classifier(cls_token_final)


def _prod3(t: Tuple[int, int, int]) -> int:
    return int(t[0]) * int(t[1]) * int(t[2])


def _has_bias(mod: nn.Module) -> bool:
    if hasattr(mod, "bias"):
        return getattr(mod, "bias") is not None
    if hasattr(mod, "layers") and len(mod.layers) > 0:
        return getattr(mod.layers[0], "bias", None) is not None
    return False


def _tdle_level(mod: nn.Module) -> int:
    return len(mod.layers) if hasattr(mod, "layers") else 1


def _flops_tensor_layer_instance(
    layer: nn.Module,
    tensor_in: Tuple[int, int, int, int, int, int],
    ignore_modes: Iterable[int],
) -> int:
    B, P1, P2, d1, d2, d3 = map(int, tensor_in)
    B_eff = B * P1 * P2

    if layer.__class__.__name__ == "TP":
        out_size = tuple(int(v) for v in layer.output_size)
        rank = tuple(int(v) for v in layer.rank)
        parts = estimate_tp_flops(
            input_size=tensor_in,
            output_size=out_size,
            rank=rank,
            ignore_modes=ignore_modes,
            include_bias=_has_bias(layer),
        )
        return int(parts["total"])

    if layer.__class__.__name__ == "TLE":
        out_rank = tuple(int(v) for v in layer.rank)
        proj = tle_input_projector_flops(tensor_in, out_rank, ignore_modes)
        b = bias_add_flops(B_eff, out_rank) if _has_bias(layer) else 0
        return int(proj + b)

    if layer.__class__.__name__ == "TDLE":
        r = _tdle_level(layer)
        out_rank = tuple(int(v) for v in layer.layers[0].rank)
        proj = tle_input_projector_flops(tensor_in, out_rank, ignore_modes)
        b = bias_add_flops(B_eff, out_rank) if _has_bias(layer) else 0
        per = int(proj + b)
        sum_cost = (r - 1) * B_eff * _prod3(out_rank)
        return int(r * per + sum_cost)

    raise ValueError(f"Unknown tensor layer type: {layer.__class__.__name__}")


def analytic_tensor_vit_flops(model: VisionTransformer) -> Tuple[float, dict]:
    """
    Analytic GFLOPs for the tensorized ViT:
      patch embed + pos-add + sum(encoder blocks) + final norm + classifier.
    Returns (gflops_total, breakdown).
    """
    B, C, H, W = model.input_size
    ps = int(model.patch_size)
    P_h, P_w = H // ps, W // ps
    d1, d2, d3 = model.embed_dim
    seq = (P_h + 1) * P_w
    dim_flat = d1 * d2 * d3

    pe = model.patch_embedding
    pe_ignore = pe.ignore_modes
    tensor_in_pe = (B, P_h, P_w, C, ps, ps)
    if pe.tensor_method == "tp":
        out_size = pe.embed_dim
        rank = tuple(int(v) for v in getattr(pe.tensor_layer, "rank"))
        pe_flops = estimate_tp_flops(
            input_size=tensor_in_pe,
            output_size=out_size,
            rank=rank,
            ignore_modes=pe_ignore,
            include_bias=_has_bias(pe.tensor_layer),
        )["total"]
    elif pe.tensor_method == "tle":
        pe_flops = tle_input_projector_flops(tensor_in_pe, pe.embed_dim, pe_ignore)
        if _has_bias(pe.tensor_layer):
            pe_flops += bias_add_flops(B * P_h * P_w, pe.embed_dim)
    elif pe.tensor_method == "tdle":
        r = _tdle_level(pe.tensor_layer)
        proj = tle_input_projector_flops(tensor_in_pe, pe.embed_dim, pe_ignore)
        b = bias_add_flops(B * P_h * P_w, pe.embed_dim) if _has_bias(pe.tensor_layer) else 0
        per = int(proj + b)
        pe_flops = int(r * per + (r - 1) * (B * P_h * P_w) * _prod3(pe.embed_dim))
    else:
        raise ValueError(f"Unknown patch method: {pe.tensor_method}")

    pos_add = B * seq * dim_flat

    enc_total = 0
    for enc in model.transformer:
        attn = enc.attention
        att_ignore = attn.ignore_modes
        h1, h2, h3 = attn.num_heads
        d1e, d2e, d3e = attn.embed_dim
        seq_e = seq
        heads_total = h1 * h2 * h3
        dph = (d1e // h1) * (d2e // h2) * (d3e // h3)
        tensor_in_block = (B, P_h + 1, P_w, d1e, d2e, d3e)

        fl_q = _flops_tensor_layer_instance(attn.tensor_layer_Q, tensor_in_block, att_ignore)
        fl_k = _flops_tensor_layer_instance(attn.tensor_layer_K, tensor_in_block, att_ignore)
        fl_v = _flops_tensor_layer_instance(attn.tensor_layer_V, tensor_in_block, att_ignore)

        fl_qk = 2 * B * heads_total * seq_e * seq_e * dph
        fl_av = 2 * B * heads_total * seq_e * seq_e * dph

        fl_out = 0
        if attn.tensor_layer_out is not None:
            fl_out = _flops_tensor_layer_instance(attn.tensor_layer_out, tensor_in_block, att_ignore)

        fl_ln = 2 * layernorm_flops(B, seq_e, dim_flat)
        fl_res = 2 * residual_add_flops(B, seq_e, dim_flat)

        fl_dp = 0
        if hasattr(enc, "drop_path") and isinstance(enc.drop_path, nn.Module):
            if getattr(enc.drop_path, "drop_prob", 0.0) > 0.0:
                fl_dp = 2 * droppath_flops(B, seq_e, dim_flat)

        mlp_l1 = enc.mlp[0]
        mlp_l2 = enc.mlp[3]
        fl_m1 = _flops_tensor_layer_instance(mlp_l1, tensor_in_block, enc.ignore_modes)
        tensor_in_hidden = (B, P_h + 1, P_w, enc.mlp_dim[0], enc.mlp_dim[1], enc.mlp_dim[2])
        fl_m2 = _flops_tensor_layer_instance(mlp_l2, tensor_in_hidden, enc.ignore_modes)

        block_total = (fl_q + fl_k + fl_v + fl_qk + fl_av + fl_out +
                       fl_ln + fl_res + fl_dp + fl_m1 + fl_m2)
        enc_total += block_total

    block_avg = enc_total // max(1, len(model.transformer))
    final_norm = layernorm_flops(B, seq, dim_flat)

    clf = model.classifier
    clf_parts = estimate_tp_flops(
        input_size=(B, d1, d2, d3),
        output_size=(model.num_classes,),
        rank=tuple(int(v) for v in clf.rank),
        ignore_modes=(0,),
        include_bias=_has_bias(clf),
    )
    clf_flops = clf_parts["total"]

    total = pe_flops + pos_add + enc_total + final_norm + clf_flops
    breakdown = {
        "patch_embed": int(pe_flops),
        "pos_add": int(pos_add),
        "encoders_total": int(enc_total),
        "encoder_avg": int(block_avg),
        "final_norm": int(final_norm),
        "classifier": int(clf_flops),
    }
    return to_gflops(total), breakdown


def analytic_report(model: VisionTransformer) -> None:
    gtot, br = analytic_tensor_vit_flops(model)
    print(f"Analytic GFLOPs (forward, batch={model.input_size[0]}): {gtot:.3f}")
    print(f"  - patch_embed:   {to_gflops(br['patch_embed']):.3f} GFLOPs")
    print(f"  - pos_add:       {to_gflops(br['pos_add']):.3f} GFLOPs")
    print(f"  - encoders:      {to_gflops(br['encoders_total']):.3f} GFLOPs (avg {to_gflops(br['encoder_avg']):.3f})")
    print(f"  - final_norm:    {to_gflops(br['final_norm']):.3f} GFLOPs")
    print(f"  - classifier:    {to_gflops(br['classifier']):.3f} GFLOPs")


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    warmup: int = 5,
    iters: int = 20,
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
    steps: int = 10,
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


def param_breakdown(vit: VisionTransformer) -> Tuple[int, int, int, int, int, int]:
    total_params, trainable_params = param_counts(vit)
    pe_params = param_counts(vit.patch_embedding)[0]
    cls_params = param_counts(vit.classifier)[0]
    enc_params = sum(param_counts(b)[0] for b in vit.transformer)
    other_params = total_params - (pe_params + cls_params + enc_params)
    return total_params, trainable_params, pe_params, enc_params, cls_params, other_params


def sanity_once(
    input_size: Tuple[int, int, int, int],
    patch_size: int,
    num_classes: int,
    embed_dim: Tuple[int, int, int],
    num_heads: Tuple[int, int, int],
    num_layers: int,
    mlp_dim: Tuple[int, int, int],
    tensor_method: str,
    tensor_method_mlp: Tuple[str, str],
    bias: bool = True,
    out_embed: bool = True,
    drop_path: float = 0.1,
    device: Optional[str] = None,
) -> None:
    """
    One full sanity + analytics pass for a given tensor/MLP method config.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vit = VisionTransformer(
        input_size=input_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        bias=bias,
        out_embed=out_embed,
        drop_path=drop_path,
        tensor_method=tensor_method,
        tensor_method_mlp=tensor_method_mlp,
    ).to(dev)

    B, C, H, W = input_size
    x = torch.randn(B, C, H, W, device=dev)
    y = vit(x)
    assert y.shape == (B, num_classes)

    total, trainable, pe, enc, clf, other = param_breakdown(vit)
    print(f"[ViT/{tensor_method}|MLP:{tensor_method_mlp}] Params: total={total/1e6:.3f}M, trainable={trainable/1e6:.3f}M")
    print(f"[ViT/{tensor_method}|MLP:{tensor_method_mlp}]   breakdown -> patch_embed={pe/1e6:.3f}M, encoders={enc/1e6:.3f}M, classifier={clf/1e6:.3f}M, other={other/1e6:.3f}M")

    analytic_report(vit)

    gflops_thop = try_thop_gflops(vit, input_size)
    print(f"[ViT/{tensor_method}|MLP:{tensor_method_mlp}] THOP GFLOPs: {'n/a' if gflops_thop is None else f'{float(gflops_thop):.2f}'}")

    lat_ms, tput = benchmark_inference(vit, input_size, warmup=5, iters=20)
    print(f"[ViT/{tensor_method}|MLP:{tensor_method_mlp}] Inference: latency={lat_ms:.2f} ms/img, throughput={tput:.1f} img/s")

    step_ms = train_one_epoch_synthetic(vit, input_size, num_classes, steps=10, lr=1e-3)
    print(f"[ViT/{tensor_method}|MLP:{tensor_method_mlp}] Synthetic train: avg step={step_ms:.2f} ms")


def sanity_suite() -> None:
    """
    Run the full suite across all tensor_method and tensor_method_mlp combinations.
    """
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    B, C, H, W = 2, 3, 32, 32
    ps = 4
    num_classes = 10
    embed_dim = (4, 4, 4)
    heads = (2, 2, 2)
    mlp_dim = (4, 4, 8)
    num_layers = 6

    methods = ("tle", "tdle", "tp")
    mlp_pairs = tuple((a, b) for a in methods for b in methods)

    for method in methods:
        for mlp_pair in mlp_pairs:
            sanity_once(
                input_size=(B, C, H, W),
                patch_size=ps,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=heads,
                num_layers=num_layers,
                mlp_dim=mlp_dim,
                tensor_method=method,
                tensor_method_mlp=mlp_pair,
            )


if __name__ == "__main__":
    sanity_suite()
