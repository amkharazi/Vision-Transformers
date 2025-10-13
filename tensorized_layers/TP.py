import sys
import time
from typing import Iterable, Optional, Sequence

sys.path.append(".")

import torch
import torch.nn as nn
from tensorized_layers.TLE import TLE
from utils.num_param import param_counts
from utils.flops import estimate_tp_flops, to_gflops, linear_flops


class TP(nn.Module):
    """
    Tensorized Projection (TP) layer.

    Notes
    -----
    The first mode of `input_size` is the batch dimension B, e.g., `(B, C, H, W)`.

    Parameters
    ----------
    input_size : Sequence[int]
        Full input shape including batch.
    output_size : Sequence[int]
        Output shape excluding batch.
    rank : Sequence[int]
        Tensor ranks; length must be `(len(input_size) - len(ignore_modes)) + len(output_size)`.
    ignore_modes : Iterable[int], default=(0,)
        Modes to ignore in the input projector (0 means batch).
    bias : bool, default=True
        If True, adds a learnable bias of shape `output_size`.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        output_size: Sequence[int],
        rank: Sequence[int],
        ignore_modes: Iterable[int] = (0,),
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_size = tuple(input_size)
        self.output_size = tuple(output_size)
        self.ignore_modes = tuple(ignore_modes)
        self.rank = tuple(rank)

        self.n = len(self.input_size) - len(self.ignore_modes)
        self.m = len(self.output_size)

        assert len(self.rank) == self.n + self.m

        self.input_rank = tuple(self.rank[: self.n])
        self.output_rank = tuple(self.rank[self.n :])

        self.input_projector = TLE(
            self.input_size, self.input_rank, ignore_modes=self.ignore_modes, bias=False
        )
        self.core = nn.Parameter(torch.randn(*self.input_rank, *self.output_rank))

        core_shape = self.input_rank + self.output_rank
        self.core_projector = TLE(
            core_shape, self.output_size, ignore_modes=tuple(range(self.n)), bias=False
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(*self.output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(B, *input_size[1:])`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(B, *output_size)`.
        """
        if self.bias is None:
            return torch.tensordot(
                self.input_projector(x),
                self.core_projector(self.core),
                dims=self.n,
            )
        return torch.tensordot(
            self.input_projector(x),
            self.core_projector(self.core),
            dims=self.n,
        ) + self.bias



def _sanity_check_tp_once(
    input_size: Sequence[int],
    output_size: Sequence[int],
    rank: Sequence[int],
    ignore_modes: Iterable[int] = (0,),
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run a TP shape, parameter-count, FLOPs, and timing check.
    """
    device = device if device is not None else torch.device("cpu")

    model = TP(
        input_size=input_size,
        output_size=output_size,
        rank=rank,
        ignore_modes=ignore_modes,
        bias=bias,
    ).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(model)
    print(f"[TP] Parameters: total={total_params:,}, trainable={trainable_params:,}")

    x = torch.randn(*input_size, device=device, dtype=dtype, requires_grad=False)

    for _ in range(warmup):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    assert tuple(y.shape) == (input_size[0],) + tuple(output_size)

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[TP] Input shape:  {tuple(input_size)}")
    print(f"[TP] Output shape: {tuple(y.shape)}")
    print(f"[TP] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    fl = estimate_tp_flops(input_size, output_size, rank, ignore_modes, include_bias=bias)
    print(f"[TP] FLOPs input_projector: {to_gflops(fl['input_projector']):.3f} GFLOPs")
    print(f"[TP] FLOPs tensordot:       {to_gflops(fl['tensordot']):.3f} GFLOPs")
    print(f"[TP] FLOPs core_projector:  {to_gflops(fl['core_projector']):.3f} GFLOPs")
    if bias:
        print(f"[TP] FLOPs bias:            {to_gflops(fl['bias']):.3f} GFLOPs")
    print(f"[TP] FLOPs TOTAL:           {to_gflops(fl['total']):.3f} GFLOPs")


def _sanity_check_linear_once(
    input_size: Sequence[int],
    output_size: Sequence[int],
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run a baseline Linear shape/params/FLOPs/timing check on flattened tensors.
    """
    device = device if device is not None else torch.device("cpu")

    B = int(input_size[0])
    Z_in = int(torch.tensor(input_size[1:]).prod().item())
    Z_out = int(torch.tensor(output_size).prod().item())

    model = nn.Linear(Z_in, Z_out, bias=bias).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(model)
    print(f"[Linear] Parameters: total={total_params:,}, trainable={trainable_params:,}")

    x = torch.randn(B, Z_in, device=device, dtype=dtype, requires_grad=False)

    for _ in range(warmup):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    assert tuple(y.shape) == (B, Z_out)

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[Linear] Input shape:  {(B, Z_in)}")
    print(f"[Linear] Output shape: {(B, Z_out)}")
    print(f"[Linear] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    fl = linear_flops(B, Z_in, Z_out, include_bias=bias)
    print(f"[Linear] FLOPs TOTAL: {to_gflops(fl):.3f} GFLOPs")


def sanity_check() -> None:
    """
    Run TP and Linear sanity checks on the same shapes.
    """
    input_size = (1024, 3, 32, 32)
    output_size = (64, 16, 16)
    ignore_modes = (0,)
    rank = (4, 8, 6, 12, 5, 7)

    _sanity_check_tp_once(input_size, output_size, rank, ignore_modes, bias=True)
    _sanity_check_tp_once(input_size, output_size, rank, ignore_modes, bias=False)
    _sanity_check_linear_once(input_size, output_size, bias=True)
    _sanity_check_linear_once(input_size, output_size, bias=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
