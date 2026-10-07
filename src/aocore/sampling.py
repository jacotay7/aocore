"""Flux-conserving resampling helpers."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from .backend import _cupy

__all__ = ["block_mean", "block_sum"]

#: Largest per-axis factor that NumPy sums by strided adds rather than a
#: single-axis reduction (measured in benchmarks/bench_block_sum.py).
_STRIDED_MAX = 4


def _factors(factor: int | tuple[int, int]) -> tuple[int, int]:
    fy, fx = (factor, factor) if isinstance(factor, int) else (int(factor[0]), int(factor[1]))
    if fy < 1 or fx < 1:
        raise ValueError("factor must be >= 1")
    return fy, fx


def _strided_sum(array: Any, factor: int, axis: int, dtype: Any) -> Any:
    """Sum every ``factor >= 2`` consecutive entries along ``axis`` by strided adds."""
    index: list[Any] = [slice(None)] * array.ndim

    def part(k: int) -> Any:
        index[axis] = slice(k, None, factor)
        return array[tuple(index)]

    if array.dtype == dtype:
        out = part(0) + part(1)
    else:  # promote first, so small integers cannot overflow
        out = part(0).astype(dtype)
        out += part(1)
    for k in range(2, factor):
        out += part(k)
    return out


def _reduce_sum(array: Any, factor: int, axis: int) -> Any:
    """Sum every ``factor`` consecutive entries along ``axis`` (-1 or -2) by reshaping."""
    shape = array.shape
    if axis == -1:
        return array.reshape(*shape[:-1], shape[-1] // factor, factor).sum(axis=-1)
    return array.reshape(*shape[:-2], shape[-2] // factor, factor, shape[-1]).sum(axis=-2)


@functools.lru_cache(maxsize=1)
def _gpu_kernel() -> Any:
    """CuPy kernel: one thread per output pixel sums its ``fy x fx`` block in row order."""
    cupy = _cupy()
    return cupy.ElementwiseKernel(
        "raw T a, int64 fy, int64 fx, int64 ny, int64 nx",
        "U out",
        """
        const long long my = ny / fy, mx = nx / fx;
        const long long plane = i / (my * mx), rem = i % (my * mx);
        const long long base = plane * ny * nx + (rem / mx) * fy * nx + (rem % mx) * fx;
        U total = (U)a[base];
        for (long long dx = 1; dx < fx; ++dx) total += (U)a[base + dx];
        for (long long dy = 1; dy < fy; ++dy) {
            for (long long dx = 0; dx < fx; ++dx) total += (U)a[base + dy * nx + dx];
        }
        out = total;
        """,
        "aocore_block_sum",
    )


def block_sum(array: Any, factor: int | tuple[int, int]) -> Any:
    """Sum ``factor x factor`` pixel blocks over the last two axes (flux-conserving binning).

    Works on NumPy and CuPy arrays alike, and returns the same dtype as
    ``array.sum()`` (bool and small integer types are promoted as NumPy
    promotes them, so sums cannot overflow where ``sum`` would not). Each
    axis length must be a multiple of its factor. Use it to integrate an
    oversampled image over detector pixels. ``factor`` is an int or a
    ``(fy, fx)`` pair; a factor of 1 returns ``array`` itself.

    Notes
    -----
    Integer sums are exact. Floating-point sums may differ from
    ``array.reshape(..., ny // fy, fy, nx // fx, fx).sum(axis=(-3, -1))`` in the
    last bits, because the order of the additions differs:

    * NumPy reduces each axis separately, rows first. Factors up to 4 use
      strided adds (``a[..., 0::f, :] + a[..., 1::f, :] + ...``), larger ones a
      single-axis sum. Both are several times faster than one reduction over
      two non-adjacent axes.
    * CuPy runs one kernel in which each thread adds its block's ``fy * fx``
      pixels in row order (a non-contiguous input is first made contiguous).

    Each output pixel is still the sum of the same ``fy * fx`` inputs, so
    float results agree to rounding (a few ulp of the block sum).
    """
    fy, fx = _factors(factor)
    ny, nx = array.shape[-2:]
    if ny % fy or nx % fx:
        raise ValueError(f"shape {array.shape[-2:]} is not a multiple of the factor ({fy}, {fx})")
    if fy == fx == 1:
        return array
    # The accumulator dtype of ``sum``: bool and small integers are promoted.
    dtype = np.add.reduce(np.zeros(1, dtype=array.dtype)).dtype
    cupy = _cupy()
    if cupy is not None and isinstance(array, cupy.ndarray):
        out = cupy.empty((*array.shape[:-2], ny // fy, nx // fx), dtype=dtype)
        return _gpu_kernel()(cupy.ascontiguousarray(array), fy, fx, ny, nx, out)
    out = array
    for f, axis in ((fy, -2), (fx, -1)):
        if f == 1:
            continue
        if f <= _STRIDED_MAX:
            out = _strided_sum(out, f, out.ndim + axis, dtype)
        else:
            out = _reduce_sum(out, f, axis)
    return out


def block_mean(array: Any, factor: int | tuple[int, int]) -> Any:
    """Average ``factor x factor`` pixel blocks over the last two axes.

    :func:`block_sum` divided by ``fy * fx``: the mean value per block, for
    resampling a transmission or an OPD map rather than integrating flux.
    Integer input gives a floating-point result (true division).
    """
    fy, fx = _factors(factor)
    return block_sum(array, (fy, fx)) / (fy * fx)
