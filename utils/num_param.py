import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def param_counts(model: nn.Module) -> tuple[int, int]:
    """
    Normalizes utils.num_param.count_parameters which may return either:
      - total (int), or
      - (total, trainable)
    """
    total_or_tuple = count_parameters(model)
    if isinstance(total_or_tuple, (tuple, list)) and len(total_or_tuple) >= 2:
        total, trainable = int(total_or_tuple[0]), int(total_or_tuple[1])
    else:
        total = int(total_or_tuple)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
