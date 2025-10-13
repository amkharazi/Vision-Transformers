import sys
import time
import math
from typing import Iterable, Optional, Sequence, Union

sys.path.append(".")

import torch
import torch.nn as nn

from tensorized_layers.TLE import TLE
from utils.num_param import param_counts
from utils.flops import tle_input_projector_flops, bias_add_flops, to_gflops, linear_flops


class TDLE(nn.Module):
    """
    Tensorized Deep Linear Encoder as a sum of r independent TLE blocks.

    Parameters
    ----------
    input_size : Sequence[int]
        Full input shape including batch, e.g. (B, C, H, W).
    rank : Union[Sequence[int], int]
        Target sizes for projected modes in each TLE block.
    ignore_modes : Iterable[int], default=(0,)
        Modes to skip in each TLE (0 is batch).
    bias : bool, default=True
        If True, each TLE has a learnable bias.
    r : int, default=3
        Number of TLE blocks summed in parallel.
    """

    def __init__(
        self,
        input_size: Sequence[int],
        rank: Union[Sequence[int], int],
        ignore_modes: Iterable[int] = (0,),
        bias: bool = True,
        r: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TLE(input_size, rank, ignore_modes, bias) for _ in range(int(r))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sum of TLE outputs.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `input_size`.

        Returns
        -------
        torch.Tensor
            Tensor matching the TLE output shape.
        """
        return sum(layer(x) for layer in self.layers)


def _tdle_output_size(input_size: Sequence[int], rank: Union[Sequence[int], int], ignore_modes: Iterable[int]) -> tuple[int, ...]:
    """
    Output shape after a single TLE given `input_size` and `rank`.
    """
    out = list(int(v) for v in input_size)
    if not isinstance(rank, (tuple, list)):
        rank = (int(rank),)
    pm = [i for i in range(len(out)) if i not in set(int(i) for i in ignore_modes)]
    for i, m in enumerate(pm):
        out[m] = int(rank[i])
    return tuple(out)


def _sanity_check_tdle_once(
    input_size: Sequence[int],
    rank: Union[Sequence[int], int],
    ignore_modes: Iterable[int] = (0,),
    bias: bool = True,
    r: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Run TDLE shape, parameter-count, FLOPs, and timing check.

    Parameters
    ----------
    input_size : Sequence[int]
        Full input shape including batch.
    rank : Union[Sequence[int], int]
        Target sizes for projected modes.
    ignore_modes : Iterable[int], default=(0,)
        Modes to skip (0 is batch).
    bias : bool, default=True
        Whether each TLE includes bias.
    r : int, default=3
        Number of TLE blocks summed in parallel.
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
    model = TDLE(input_size=input_size, rank=rank, ignore_modes=ignore_modes, bias=bias, r=r).to(device=device, dtype=dtype)

    total_params, trainable_params = param_counts(model)
    print(f"[TDLE] Parameters: total={total_params:,}, trainable={trainable_params:,}")

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

    expected = _tdle_output_size(input_size, rank, ignore_modes)
    assert tuple(y.shape) == expected

    avg_ms = (sum(times) / len(times)) * 1000.0
    print(f"[TDLE] Input shape:  {tuple(input_size)}")
    print(f"[TDLE] Output shape: {tuple(y.shape)}")
    print(f"[TDLE] Avg forward time over {iters} iters (warmup {warmup}): {avg_ms:.3f} ms")

    B = int(input_size[0])
    out_no_batch = expected[1:]
    if not isinstance(rank, (tuple, list)):
        rank = (int(rank),)
    fl_tle = tle_input_projector_flops(input_size, rank, ignore_modes)
    fl_bias = bias_add_flops(B, out_no_batch) if bias else 0
    fl_sum = (int(r) - 1) * B * math.prod(out_no_batch)
    total = int(r) * (fl_tle + fl_bias) + fl_sum

    print(f"[TDLE] FLOPs per TLE projector: {to_gflops(fl_tle):.3f} GFLOPs")
    if bias:
        print(f"[TDLE] FLOPs per TLE bias:      {to_gflops(fl_bias):.3f} GFLOPs")
    print(f"[TDLE] FLOPs output sum:         {to_gflops(fl_sum):.3f} GFLOPs")
    print(f"[TDLE] FLOPs TOTAL:              {to_gflops(total):.3f} GFLOPs")


def _sanity_check_linear_once(
    input_size: Sequence[int],
    rank: Union[Sequence[int], int],
    ignore_modes: Iterable[int] = (0,),
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    warmup: int = 2,
    iters: int = 5,
) -> None:
    """
    Baseline Linear on flattened tensors matching TDLE output size.
    """
    device = device if device is not None else torch.device("cpu")
    B = int(input_size[0])
    Z_in = int(torch.tensor(input_size[1:]).prod().item())
    Z_out = int(torch.tensor(_tdle_output_size(input_size, rank, ignore_modes)[1:]).prod().item())

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
    Run TDLE and baseline Linear sanity checks.
    """
    input_size = (1024, 3, 32, 32)
    rank = (4, 8, 6)
    ignore_modes = (0,)
    r = 3

    _sanity_check_tdle_once(input_size, rank, ignore_modes, bias=True, r=r)
    _sanity_check_tdle_once(input_size, rank, ignore_modes, bias=False, r=r)
    _sanity_check_linear_once(input_size, rank, ignore_modes, bias=True)
    _sanity_check_linear_once(input_size, rank, ignore_modes, bias=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    sanity_check()
