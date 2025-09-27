import sys
import math
import time
from typing import Iterable, Optional, Sequence, Union

sys.path.append(".")

import torch
import torch.nn as nn
from torch.nn import init

from utils.num_param import param_counts
from utils.flops import (
    tle_input_projector_flops,
    bias_add_flops,
    to_gflops,
    linear_flops,
)


class TLE(nn.Module):
    """
    Tensorized Linear Encoder.

    Parameters
    ----------
    input_size : Union[Sequence[int], int]
        Full input shape including batch, e.g. (B, C, H, W).
    rank : Union[Sequence[int], int]
        Target sizes for projected modes, one per non-ignored mode.
    ignore_modes : Iterable[int], default=(0,)
        Modes to skip (0 is batch).
    bias : bool, default=True
        If True, adds learnable bias broadcastable to output.
    """

    def __init__(
        self,
        input_size: Union[Sequence[int], int],
        rank: Union[Sequence[int], int],
        ignore_modes: Iterable[int] = (0,),
        bias: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(input_size, int):
            input_size = (input_size,)
        self.input_size = tuple(int(x) for x in input_size)
        self.ignore_modes = set(int(i) for i in ignore_modes)
        self.rank = tuple(int(r) for r in (rank if isinstance(rank, (tuple, list)) else (rank,)))

        self.project_modes = [i for i in range(len(self.input_size)) if i not in self.ignore_modes]
        assert len(self.project_modes) == len(self.rank)

        self.linears = nn.ModuleDict(
            {
                str(mode): nn.Linear(self.input_size[mode], self.rank[i], bias=False)
                for i, mode in enumerate(self.project_modes)
            }
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(*self.rank))
        else:
            self.register_parameter("bias", None)

        self.init_parameters()

    def init_parameters(self) -> None:
        """Kaiming-uniform weights; uniform bias scaled by first projected fan-in."""
        for linear in self.linears.values():
            init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.input_size[self.project_modes[0]]
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequential mode projections by permute-reshape-linear-reshape-inverse-permute.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `input_size`.

        Returns
        -------
        torch.Tensor
            Tensor whose shape matches `input_size` with projected modes replaced by `rank`.
        """
        for i, mode in enumerate(self.project_modes):
            perm = list(range(x.ndim))
            perm[mode], perm[-1] = perm[-1], perm[mode]
            inv_perm = [0] * len(perm)
            for j, p in enumerate(perm):
                inv_perm[p] = j
            s = x.permute(perm).shape
            x = self.linears[str(mode)](x.permute(perm).reshape(-1, s[-1])).reshape(*s[:-1], self.rank[i]).permute(inv_perm)

        if self.bias is None:
            return x
        return x + self.bias


def _tle_output_size(input_size: Sequence[int], rank: Sequence[int], ignore_modes: Iterable[int]) -> tuple[int, ...]:
    """Output shape after TLE given `input_size` and `rank`."""
    out = list(int(v) for v in input_size)
    project_modes = [i for i in range(len(out)) if i not in set(int(i) for i in ignore_modes)]
    for i, m in enumerate(project_modes):
        out[m] = int(rank[i])
    return tuple(out)


def _sanity_check_once(
    input_size: Sequence[int],
    rank: Sequence[int],
    ignore_modes: Iterable[int] = (0,),
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Check TLE shape/params/time/FLOPs.

    Parameters
    ----------
    input_size : Sequence[int]
        Full input shape including batch (e.g., (B, C, H, W)).
    rank : Sequence[int]
        Target sizes for projected modes.
    ignore_modes : Iterable[int], default=(0,)
        Modes to skip (0 is batch).
    bias : bool, default=True
        Whether to include bias.
    device : torch.device, optional
        Device to use.
    dtype : torch.dtype, default=torch.float32
        Tensor dtype.
    warmup : int, default=2
        Warmup forward passes.
    iters : int, default=5
        Timed forward passes to average.
    """
    device = device if device is not None else torch.device("cpu")
    model = TLE(input_size=input_size, rank=rank, ignore_modes=ignore_modes, bias=bias).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(model)
    print(f"[TLE] Parameters: total={total_params:,}, trainable={trainable_params:,}")

    x = torch.randn(*input_size, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    expected = _tle_output_size(input_size, rank, ignore_modes)
    assert tuple(y.shape) == expected

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[TLE] Input shape:  {tuple(input_size)}")
    print(f"[TLE] Output shape: {tuple(y.shape)}")
    print(f"[TLE] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    B = int(input_size[0])
    out_no_batch = expected[1:]
    fl_proj = tle_input_projector_flops(input_size, rank, ignore_modes)
    fl_bias = bias_add_flops(B, out_no_batch) if bias else 0
    print(f"[TLE] FLOPs projectors: {to_gflops(fl_proj):.3f} GFLOPs")
    if bias:
        print(f"[TLE] FLOPs bias:       {to_gflops(fl_bias):.3f} GFLOPs")
    print(f"[TLE] FLOPs TOTAL:      {to_gflops(fl_proj + fl_bias):.3f} GFLOPs")


def _sanity_check_linear_once(
    input_size: Sequence[int],
    rank: Sequence[int],
    ignore_modes: Iterable[int] = (0,),
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Baseline Linear on flattened tensors matching TLE's input/output sizes.
    """
    device = device if device is not None else torch.device("cpu")
    B = int(input_size[0])
    Z_in = int(torch.tensor(input_size[1:]).prod().item())
    Z_out = int(torch.tensor(_tle_output_size(input_size, rank, ignore_modes)[1:]).prod().item())

    model = nn.Linear(Z_in, Z_out, bias=bias).to(device=device, dtype=dtype)
    total_params, trainable_params = param_counts(model)
    print(f"[Linear] Parameters: total={total_params:,}, trainable={trainable_params:,}")

    x = torch.randn(B, Z_in, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times: list[float] = []
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
    Run TLE and baseline Linear sanity checks.
    """
    input_size = (1024, 3, 32, 32)
    rank = (4, 8, 6)
    ignore_modes = (0,)
    _sanity_check_once(input_size, rank, ignore_modes, bias=True)
    _sanity_check_once(input_size, rank, ignore_modes, bias=False)
    _sanity_check_linear_once(input_size, rank, ignore_modes, bias=True)
    _sanity_check_linear_once(input_size, rank, ignore_modes, bias=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
