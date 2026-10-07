"""Flux-conserving resampling helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["block_sum"]


def block_sum(array: Any, factor: int | tuple[int, int]) -> Any:
    """Sum ``factor x factor`` pixel blocks over the last two axes (flux-conserving binning).

    Works on NumPy and CuPy arrays alike. Each axis length must be a multiple of
    its factor. Use it to integrate an oversampled image over detector pixels.
    """
    fy, fx = (factor, factor) if isinstance(factor, int) else (int(factor[0]), int(factor[1]))
    if fy < 1 or fx < 1:
        raise ValueError("factor must be >= 1")
    ny, nx = array.shape[-2:]
    if ny % fy or nx % fx:
        raise ValueError(f"shape {array.shape[-2:]} is not a multiple of the factor ({fy}, {fx})")
    if fy == fx == 1:
        return array
    lead = array.shape[:-2]
    return array.reshape(*lead, ny // fy, fy, nx // fx, fx).sum(axis=(-3, -1))
