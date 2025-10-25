import sys

sys.path.append(".")

import torch
import torch.nn as nn
from utils.n_mode_product import n_mode_product_einsum
from utils.num_param import count_parameters


class TP(nn.Module):
    def __init__(self, input_size, output_size, ranks, ignore_modes=(0,), bias=True):
        super().__init__()
        self.input_size = tuple(input_size)
        self.output_size = tuple(output_size)
        self.ignore_modes = tuple(ignore_modes)
        self.ranks = tuple(ranks)

        self.project_axes = tuple(
            i for i in range(len(self.input_size)) if i not in self.ignore_modes
        )
        n = len(self.input_size) - len(self.ignore_modes)

        rin = self.ranks[:n]
        rout = self.ranks[n:]

        self.register_parameter("G", nn.Parameter(torch.empty(*self.ranks)))
        nn.init.normal_(self.G, mean=0.0, std=0.02)

        for ax, r in zip(self.project_axes, rin):
            I = self.input_size[ax]
            P = nn.Parameter(torch.empty(r, I))
            nn.init.xavier_uniform_(P)
            self.register_parameter(f"U_{ax}", P)

        for j, (O, r) in enumerate(zip(self.output_size, rout)):
            P = nn.Parameter(torch.empty(r, O))
            nn.init.xavier_uniform_(P)
            self.register_parameter(f"V_{j}", P)

        if bias:
            b = nn.Parameter(torch.empty(*self.output_size))
            nn.init.normal_(b, mean=0.0, std=0.02)
            self.register_parameter("bias", b)
        else:
            self.register_parameter("bias", None)

    def _apply_core_maps(self):
        n = len(self.input_size) - len(self.ignore_modes)
        m = len(self.output_size)
        Gt = self.G
        for i, ax in enumerate(self.project_axes):
            A = getattr(self, f"U_{ax}")
            Gt = n_mode_product_einsum(Gt, A, mode=i)
        for j in range(m):
            B = getattr(self, f"V_{j}")
            Gt = n_mode_product_einsum(Gt, B, mode=n + j)
        return Gt

    def forward(self, x):
        m = len(self.output_size)
        n = len(self.input_size) - len(self.ignore_modes)
        Gmapped = self._apply_core_maps()
        letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        N = x.ndim
        idx_x = list(letters[:N])
        idx_G_in = [idx_x[ax] for ax in self.project_axes]
        idx_G_out = list(letters[N : N + m])
        idx_G = idx_G_in + idx_G_out
        idx_out = idx_x.copy()[:len(self.ignore_modes)] + idx_G_out
        eq = f"{''.join(idx_x)},{''.join(idx_G)}->{''.join(idx_out)}"
        x = torch.einsum(eq, x, Gmapped)
        if self.bias is not None:
            x = x + self.bias
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    input_size = (2, 3, 4, 5, 6, 7)
    output_size = (9, 10, 11)
    ranks = (1, 2, 3, 4, 5, 6)
    ignore_modes = (0, 1, 2)

    model = TP(input_size, output_size, ranks, ignore_modes, bias=True)

    x = torch.randn(*input_size)
    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameter count:", count_parameters(model))
    
    
    input_size = (2, 3, 4, 5, 6, 7)
    output_size = (200, )
    ranks = (1,2,3,4)
    ignore_modes = (0, 1, 2)

    model = TP(input_size, output_size, ranks, ignore_modes, bias=True)

    x = torch.randn(*input_size)
    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameter count:", count_parameters(model))
    
    input_size = (2, 5, 6, 7)
    output_size = (200, )
    ranks = (1,2,3,4)
    ignore_modes = (0,)

    model = TP(input_size, output_size, ranks, ignore_modes, bias=True)

    x = torch.randn(*input_size)
    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameter count:", count_parameters(model))
