import sys

sys.path.append(".")

import torch
import torch.nn as nn
from tensorized_layers.TLE import TLE
from utils.num_param import count_parameters


class TDLE(nn.Module):
    def __init__(self, input_size, ranks, depth, bias=False, ignore_modes=(0,)):
        super().__init__()
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"`depth` must be a positive int, got {depth}")

        self.input_size = tuple(input_size)
        self.ranks = tuple(ranks)
        self.depth = depth
        self.bias = bool(bias)
        self.ignore_modes = tuple(ignore_modes)

        self.branches = nn.ModuleList(
            [
                TLE(
                    input_size=self.input_size,
                    ranks=self.ranks,
                    bias=self.bias,
                    ignore_modes=self.ignore_modes,
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return torch.stack(outs, dim=0).sum(dim=0)
    
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    import torch
    from utils.num_param import count_parameters

    torch.manual_seed(0)

    input_size = (2, 3, 4, 5, 6, 7)
    ranks = (8, 19, 12)
    ignore_modes = (0, 1, 2)
    depth = 4

    model = TDLE(
        input_size=input_size,
        ranks=ranks,
        depth=depth,
        bias=True,
        ignore_modes=ignore_modes,
    )

    x = torch.randn(*input_size)
    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Depth:", depth)
    print("Parameter count (TDLE total):", count_parameters(model))

    from tensorized_layers.TLE import TLE as _SingleTLE
    single = _SingleTLE(input_size=input_size, ranks=ranks, bias=True, ignore_modes=ignore_modes)
    print("Parameter count (single TLE):", count_parameters(single))
