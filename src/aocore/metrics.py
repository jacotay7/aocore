"""Wavefront error metrics that respect phase-retrieval ambiguities.

Phase retrieval cannot see some components of the wavefront: piston always,
absolute tip/tilt when the image position is free, and for a single in-focus
image of a centro-symmetric pupil the *twin* ``phi(x) -> -phi(-x)``. These
helpers compare an estimate with a reference after removing exactly those
components, weighted by pupil intensity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .backend import _cupy, to_numpy
from .conventions import coordinate_grid
from .pupil import Pupil

__all__ = [
    "remove_modes",
    "rms",
    "rms_tiptilt_removed",
    "rms_unweighted",
    "strehl_from_rms",
    "wavefront_error",
]

RemoveSpec = str | Sequence[str] | None


def _removal_maps(pupil: Pupil, remove: RemoveSpec) -> list[np.ndarray]:
    if remove is None:
        return []
    names = [remove] if isinstance(remove, str) else list(remove)
    y, x = pupil.coordinates()
    maps: list[np.ndarray] = []
    for name in names:
        if name == "piston":
            maps.append(np.ones(pupil.shape))
        elif name in ("tip", "tiptilt"):
            if name == "tiptilt":
                maps += [np.ones(pupil.shape), np.asarray(x), np.asarray(y)]
            else:
                maps.append(np.asarray(x))
        elif name == "tilt":
            maps.append(np.asarray(y))
        elif name == "segment_piston":
            if pupil.segments is None:
                raise ValueError("segment_piston needs a segmented pupil")
            maps += [(pupil.segments == s).astype(float) for s in range(1, pupil.n_segments + 1)]
        else:
            raise ValueError(f"unknown mode to remove {name!r}")
    return maps


def _as_pupil(pupil: Any) -> Pupil:
    return pupil if isinstance(pupil, Pupil) else Pupil.from_array(to_numpy(pupil))


def remove_modes(opd: Any, pupil: Pupil | Any, remove: RemoveSpec = "piston") -> np.ndarray:
    """Subtract the intensity-weighted least-squares fit of the named modes.

    ``remove`` is ``"piston"``, ``"tip"``, ``"tilt"``, ``"tiptilt"``
    (piston + tip + tilt), ``"segment_piston"``, a list of these, or None.
    """
    pupil = _as_pupil(pupil)
    data = np.asarray(to_numpy(opd), dtype=np.float64)
    if data.shape != pupil.shape:
        raise ValueError(f"OPD shape {data.shape} does not match the pupil {pupil.shape}")
    mask = pupil.mask
    out = np.where(mask, data, 0.0)
    maps = _removal_maps(pupil, remove)
    if not maps:
        return out
    w = np.sqrt((pupil.amplitude**2)[mask])
    basis = np.stack([m[mask] for m in maps], axis=1)
    coef, *_ = np.linalg.lstsq(basis * w[:, None], data[mask] * w, rcond=None)
    out[mask] = data[mask] - basis @ coef
    return out


def rms(opd: Any, pupil: Pupil | Any, remove: RemoveSpec = "piston") -> float:
    """Intensity-weighted RMS over the pupil after :func:`remove_modes` (CONVENTIONS 4.1).

    ``pupil`` is a :class:`~aocore.Pupil` or a pupil amplitude array.
    """
    pupil = _as_pupil(pupil)
    resid = remove_modes(opd, pupil, remove)
    w = pupil.amplitude**2
    return float(math.sqrt(np.sum(w * resid**2) / np.sum(w)))


def _namespace(*arrays: Any) -> Any:
    """CuPy if any of ``arrays`` lives on the GPU, else NumPy."""
    cupy = _cupy()
    if cupy is not None and any(isinstance(a, cupy.ndarray) for a in arrays):
        return cupy
    return np


def rms_unweighted(opd: Any, mask: Pupil | Any = None) -> float:
    """Plain quadratic mean of ``opd`` over ``mask``, piston included (CONVENTIONS 4.1 variant).

    Returns ``sqrt(mean(opd[mask] ** 2))``: every pixel inside the mask
    counts equally, whatever the pupil transmission, and **nothing is
    removed**, so a constant OPD ``c`` gives ``|c|``. This is the figure that
    packages such as makewfs report for a residual OPD that is already
    piston-free. Use :func:`rms` for the convention's intensity-weighted,
    piston-removed RMS.

    Works on NumPy and CuPy arrays without copying the OPD to the host; the
    sum is accumulated in float64 on the array's device, and only the scalar
    result crosses to the host.

    Parameters
    ----------
    opd:
        OPD map (any shape; the result is in its units). NumPy or CuPy.
    mask:
        Pixels to include: a boolean array, an amplitude-like array whose
        non-zero pixels count, or a :class:`~aocore.Pupil` (its
        :attr:`~aocore.Pupil.mask`). Same shape as ``opd``. ``None`` uses
        the whole array. Values outside the mask are ignored, even if they
        are NaN.
    """
    if isinstance(mask, Pupil):
        mask = mask.mask
    xp = _namespace(opd, mask)
    data = xp.asarray(opd, dtype=xp.float64)
    if mask is None:
        if data.size == 0:
            raise ValueError("opd is empty")
        return math.sqrt(float(xp.mean(data * data)))
    inside = xp.asarray(mask) != 0
    if inside.shape != data.shape:
        raise ValueError(f"opd {data.shape} and mask {inside.shape} differ in shape")
    count = int(xp.count_nonzero(inside))
    if count == 0:
        raise ValueError("mask is empty")
    return math.sqrt(float(xp.sum(xp.where(inside, data * data, 0.0))) / count)


def rms_tiptilt_removed(opd: Any, pupil: Pupil | Any) -> float:
    """Intensity-weighted RMS with piston, tip and tilt removed (CONVENTIONS 4.1 variant).

    The residual after subtracting the weighted least-squares fit of
    ``c0 + c1 x + c2 y``, with weights ``|pupil|^2``:
    ``sqrt(sum a^2 r^2 / sum a^2)``. Equal to ``rms(opd, pupil, "tiptilt")``,
    but it runs on the array's device: NumPy and CuPy inputs stay where they
    are, the reductions run in float64 there, and only a handful of scalars
    (the 2x2 normal equations and the result) cross to the host.

    Parameters
    ----------
    opd:
        2-D OPD map (the result is in its units). NumPy or CuPy.
    pupil:
        A :class:`~aocore.Pupil` or a pupil amplitude array of the same
        shape. Pixels with zero amplitude do not count (their OPD may be
        NaN).
    """
    amplitude = pupil.amplitude if isinstance(pupil, Pupil) else pupil
    xp = _namespace(opd, amplitude)
    data = xp.asarray(opd, dtype=xp.float64)
    weight = xp.asarray(amplitude, dtype=xp.float64) ** 2
    if data.ndim != 2 or data.shape != weight.shape:
        raise ValueError(f"opd {data.shape} and pupil {weight.shape} must be equal 2-D shapes")
    total = float(xp.sum(weight))
    if not total > 0:
        raise ValueError("pupil has no transmission")
    data = xp.where(weight > 0, data, 0.0)
    y, x = coordinate_grid(data.shape, backend="gpu" if xp is not np else "cpu", dtype=np.float64)
    # Remove the weighted means first, so the remaining 2x2 system for tip
    # and tilt is well conditioned whatever the pupil's offset or piston.
    means = to_numpy(xp.stack([xp.sum(weight * v) for v in (data, x, y)])) / total
    d, dx, dy = data - means[0], x - means[1], y - means[2]
    wdx, wdy = weight * dx, weight * dy
    sums = to_numpy(xp.stack([xp.sum(v) for v in (wdx * dx, wdx * dy, wdy * dy, wdx * d, wdy * d)]))
    normal = np.array([[sums[0], sums[1]], [sums[1], sums[2]]])
    coef = np.linalg.lstsq(normal, sums[3:], rcond=None)[0]
    resid = d - float(coef[0]) * dx - float(coef[1]) * dy
    return math.sqrt(float(xp.sum(weight * resid * resid)) / total)


def strehl_from_rms(rms_opd: float, wavelength: float) -> float:
    """Marechal approximation ``exp(-(2 pi rms / lambda)^2)`` (alias of ``strehl_marechal``)."""
    return float(math.exp(-((2.0 * math.pi * rms_opd / wavelength) ** 2)))


def wavefront_error(
    estimate: Any,
    truth: Any,
    pupil: Pupil | Any,
    *,
    remove: RemoveSpec = "piston",
    allow_twin: bool = False,
) -> float:
    """RMS difference between two OPD maps over the pupil, modulo ambiguities.

    Parameters
    ----------
    estimate, truth:
        OPD maps (same units; the result is in those units).
    remove:
        Modes removed from the difference before the RMS (see
        :func:`remove_modes`).
    allow_twin:
        Also compare against the twin ``-estimate(-x)`` and return the smaller
        error (for single-image retrieval without diversity). Only meaningful
        for pupils symmetric under 180-degree rotation.
    """
    pupil = _as_pupil(pupil)
    est = np.asarray(to_numpy(estimate), dtype=np.float64)
    ref = np.asarray(to_numpy(truth), dtype=np.float64)
    err = rms(est - ref, pupil, remove)
    if allow_twin:
        twin = -est[::-1, ::-1]
        err = min(err, rms(twin - ref, pupil, remove))
    return err
