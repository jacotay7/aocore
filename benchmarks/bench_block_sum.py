"""Benchmark ``aocore.block_sum`` against the 0.1.2 implementation.

Run from the repository root::

    python benchmarks/bench_block_sum.py          # NumPy
    python benchmarks/bench_block_sum.py --gpu    # also CuPy, if available

Prints a Markdown table of the best-of-N time per call in microseconds for
the 0.1.2 two-axis reduction (``sum(axis=(-3, -1))``), the current
``block_sum``, and, for factor 2, the strided-add fast path that makewfs kept
for itself.
"""

from __future__ import annotations

import argparse
import timeit
from collections.abc import Callable
from typing import Any

import numpy as np

import aocore as ac

CASES = [
    ((240, 240), 2),
    ((240, 240), 3),
    ((240, 240), 4),
    ((240, 240), 8),
    ((1024, 1024), 2),
    ((1024, 1024), 4),
    ((1024, 1024), 8),
    ((64, 96, 96), 2),
    ((64, 96, 96), 3),
    ((64, 96, 96), 4),
    ((8, 512, 512), 2),
    ((8, 512, 512), 4),
]


def two_axis(array: Any, f: int) -> Any:
    """aocore 0.1.2: one reduction over both block axes."""
    *lead, ny, nx = array.shape
    return array.reshape(*lead, ny // f, f, nx // f, f).sum(axis=(-3, -1))


def makewfs_factor2(array: Any) -> Any:
    """The factor-2 fast path makewfs carried before aocore 0.1.3."""
    return (
        array[..., ::2, ::2]
        + array[..., 1::2, ::2]
        + array[..., ::2, 1::2]
        + array[..., 1::2, 1::2]
    )


def best_time(fn: Callable[[], Any], sync: Callable[[], None], number: int, repeat: int) -> float:
    def run() -> None:
        fn()
        sync()

    run()
    return min(timeit.repeat(run, number=number, repeat=repeat)) / number * 1e6


HEADER = (
    "| device | dtype | shape | factor | 0.1.2 (us) | block_sum (us) | speed-up | makewfs f=2 |"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true", help="also time CuPy arrays")
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    devices = ["cpu"] + (["gpu"] if args.gpu and ac.gpu_available() else [])
    print(HEADER)
    print("|---|---|---|---|---:|---:|---:|---:|")
    rng = np.random.default_rng(0)
    for device in devices:
        for dtype in (np.float32, np.float64):
            be = ac.get_backend(device, "single" if dtype == np.float32 else "double")
            for shape, f in CASES:
                data = be.asarray(rng.random(shape).astype(dtype))
                number = max(1, int(2e6 // data.size))

                def timed(fn: Callable[[], Any], number: int = number, be: Any = be) -> float:
                    return best_time(fn, be.synchronize, number, args.repeat)

                old = timed(lambda d=data, f=f: two_axis(d, f))
                new = timed(lambda d=data, f=f: ac.block_sum(d, f))
                fast = f"{timed(lambda d=data: makewfs_factor2(d)):.1f}" if f == 2 else ""
                print(
                    f"| {device} | {np.dtype(dtype).name} | {shape} | {f} | {old:.1f} | {new:.1f} "
                    f"| {old / new:.2f}x | {fast} |"
                )


if __name__ == "__main__":
    main()
