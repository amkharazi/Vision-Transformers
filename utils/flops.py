from typing import Optional, Tuple

import torch
import torch.nn as nn


def try_flops_gflops(model: nn.Module, input_size: Tuple[int, int, int, int]) -> Optional[float]:
    """
    Estimate GFLOPs using thop.profile. Returns None if thop isn't installed or profiling fails.
    Preserves the model's train/eval state and avoids autograd side effects.
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
