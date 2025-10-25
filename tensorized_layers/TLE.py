import sys

sys.path.append(".")

import torch
import torch.nn as nn
from utils.n_mode_product import n_mode_product_einsum
from utils.num_param import count_parameters


class TLE(nn.Module):
    """
    A tensorized layer over an N-D input; input_size includes batch.
    len(ranks) + len(ignore_modes) == len(input_size)
    Factors have shape (r_i, I_i). By default ignores batch mode 0.
    Requires batch size mode unless you remove 0 from ignore_modes.
    """

    def __init__(self, input_size, ranks, bias=False, ignore_modes=(0,)):
        super().__init__()
        self.input_size = tuple(input_size)
        self.ranks = tuple(ranks)
        self.ignore_modes = tuple(ignore_modes)

        assert len(self.input_size) == len(self.ranks) + len(self.ignore_modes)

        self.transform_axes = tuple(
            ax for ax in range(len(self.input_size)) if ax not in self.ignore_modes
        )

        for ax, r in zip(self.transform_axes, self.ranks):
            I = self.input_size[ax]
            p = nn.Parameter(torch.empty(I, r))
            nn.init.xavier_uniform_(p)
            self.register_parameter(f"U_{ax}", p)

        last_m_axes = tuple(
            range(len(self.input_size) - len(self.ranks), len(self.input_size))
        )
        can_add_bias = tuple(self.transform_axes) == last_m_axes

        if bias and can_add_bias:
            b = nn.Parameter(torch.empty(*self.ranks))
            nn.init.normal_(b, mean=0.0, std=0.02)
            self.register_parameter("bias", b)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        for ax in self.transform_axes:
            U = getattr(self, f"U_{ax}")
            x = n_mode_product_einsum(x, U, mode=ax)
        if self.bias is not None:
            m = len(self.ranks)
            x = x + self.bias.view(*([1] * (x.ndim - m)), *self.ranks)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    input_size = (2, 3, 4, 5, 6, 7)
    ranks = (8, 19, 12)
    ignore_modes = (0, 1, 2)
    model = TLE(
        input_size=input_size, ranks=ranks, bias=True, ignore_modes=ignore_modes
    )

    x = torch.randn(*input_size)
    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Bias is None:", model.bias is None)
    print("Parameter count:", count_parameters(model))
