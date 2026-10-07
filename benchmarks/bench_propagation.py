"""Benchmark the propagators and ``Backend.dot`` against the 0.1.3 code paths.

Run from the repository root::

    python benchmarks/bench_propagation.py          # NumPy/SciPy
    python benchmarks/bench_propagation.py --gpu    # also CuPy, if available

Prints a Markdown table of the median time per call in microseconds for the
0.1.3 implementation (full zero-padded ``fft2`` then crop, broadcast batched
``matmul``, ``vdot``) and the current one, with two diversity channels
(``K = 2``) as a focal-plane solver uses them, and checks that both agree.
Thread counts follow the library defaults (``AOCORE_FFT_WORKERS`` and
``AOCORE_BLAS_THREADS`` override them); the 0.1.3 MFT also used 4 BLAS
threads below 2**28 multiply-adds on every machine.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

import aocore as ac
from aocore.backend import _blas_threads

FFT_SIZES = [64, 128, 256, 512]
MFT_CASES = [(128, 1), (128, 5), (256, 5)]  # (pupil size, wavelengths)


def median_us(fn: Callable[[], Any], backend: ac.Backend, repeats: int = 31) -> float:
    for _ in range(5):
        fn()
    backend.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        backend.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e6


def fft_013(prop: ac.FFTPropagator, field: Any) -> Any:
    """aocore 0.1.3 FFTPropagator.forward: pad, full fft2, crop."""
    be = prop.backend
    ny, nx = prop.in_shape
    padded = be.xp.zeros((*field.shape[:-2], *prop.n_fft), dtype=be.complex_dtype)
    padded[..., :ny, :nx] = field * prop._in_mod
    my, mx = prop.out_shape
    return be.fft2(padded)[..., :my, :mx] * prop._out_mod


def mft_013(prop: ac.MFTPropagator, field: Any) -> Any:
    """aocore 0.1.3 MFTPropagator.forward: broadcast batched matmul."""
    xp = prop.backend.xp
    work = prop._work(field)
    if not prop.backend.is_gpu and work <= 2**28:
        work = 0.0  # 0.1.3 capped mid-sized products at 4 threads everywhere
    with prop.backend.blas_limit(work):
        return xp.matmul(xp.matmul(prop._ay, field), prop._axt)


def compare(
    name: str, old: Callable[[], Any], new: Callable[[], Any], be: ac.Backend, repeats: int = 31
) -> tuple[str, float, float, float]:
    """Time both versions and return their largest difference relative to the peak."""
    reference, result = be.to_numpy(old()), be.to_numpy(new())
    diff = float(np.abs(reference - result).max() / np.abs(reference).max())
    return name, median_us(old, be, repeats), median_us(new, be, repeats), diff


def rows(device: str) -> list[tuple[str, float, float, float]]:
    be = ac.get_backend(device)
    rng = np.random.default_rng(0)
    out = []

    def field(shape: tuple[int, ...]) -> Any:
        return be.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape), "complex")

    for n in FFT_SIZES:
        fft = ac.FFTPropagator(n, n, 2 * n, backend=be)
        u = field((2, n, n))
        name = f"FFT forward {n} -> {2 * n}"
        out.append(compare(name, partial(fft_013, fft, u), partial(fft.forward, u), be))
    for n, wavelengths in MFT_CASES:
        samples = list(np.linspace(1.8, 2.2, wavelengths) * n) if wavelengths > 1 else [2.06 * n]
        mft = ac.MFTPropagator(n, n, samples, stack=True, backend=be)
        u = field((2, wavelengths, n, n))
        name = f"MFT forward {n}, L={wavelengths}"
        out.append(compare(name, partial(mft_013, mft, u), partial(mft.forward, u), be))
    a = be.asarray(rng.standard_normal(65536))
    b = be.asarray(rng.standard_normal(65536))
    xp = be.xp
    old_dot: Callable[[], Any] = partial(be.dot, a, b)
    if be.is_gpu:
        old_dot = lambda: float(xp.vdot(a, b).real)  # noqa: E731 (0.1.3 GPU dot)
    out.append(compare("dot, 65536 float", old_dot, partial(be.dot, a, b), be, 201))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true", help="also time CuPy")
    args = parser.parse_args()
    devices = ["cpu"] + (["gpu"] if args.gpu and ac.gpu_available() else [])
    print(f"BLAS threads for a 1e8 multiply-add product: {_blas_threads(1e8)}\n")
    print("| device | case | 0.1.3 (us) | now (us) | speed-up | max rel. difference |")
    print("|---|---|---:|---:|---:|---:|")
    for device in devices:
        for name, old, new, diff in rows(device):
            print(f"| {device} | {name} | {old:.0f} | {new:.0f} | {old / new:.2f}x | {diff:.1e} |")


if __name__ == "__main__":
    main()
