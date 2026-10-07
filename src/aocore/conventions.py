"""Reference implementations of the stack conventions (see CONVENTIONS.md).

Small, dependency-free definitions that the conformance checks compare
against, and that packages can import instead of re-deriving.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .backend import BackendLike, get_backend

__all__ = [
    "ARCSEC_TO_RAD",
    "RAD_TO_ARCSEC",
    "centered_coordinates",
    "centroid",
    "coordinate_grid",
    "opd_to_phase",
    "phase_to_opd",
    "rms",
    "strehl_marechal",
]

#: Radians per arcsecond (CONVENTIONS 2.3).
ARCSEC_TO_RAD = math.pi / 648000.0
#: Arcseconds per radian.
RAD_TO_ARCSEC = 648000.0 / math.pi


def centered_coordinates(
    n: int,
    pitch: float = 1.0,
    offset: float = 0.0,
    *,
    dtype: Any = None,
    backend: BackendLike = None,
) -> Any:
    """Pixel-centre coordinates ``(i - (n - 1) / 2 + offset) * pitch`` (CONVENTIONS 1.2).

    The values are computed in float64 and then cast, so a float32 grid
    equals the float32 cast of the float64 one on either device.

    Parameters
    ----------
    n:
        Number of pixels along the axis.
    pitch:
        Pixel pitch (any unit; metres for pupil planes).
    offset:
        Shift of the grid in pixels.
    dtype:
        Floating-point dtype of the result. Defaults to the backend's working
        precision (:attr:`Backend.real_dtype`), which is float64 for the
        default CPU backend.
    backend:
        Where to build the array: a :class:`~aocore.Backend` or a device name
        (see :func:`~aocore.get_backend`). Default: host NumPy, double
        precision.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    be = get_backend(backend)
    xp = be.xp
    coords = (xp.arange(n, dtype=xp.float64) - (n - 1) / 2.0 + offset) * pitch
    return coords.astype(be.real_dtype if dtype is None else dtype, copy=False)


def coordinate_grid(
    shape: Any,
    pitch: float = 1.0,
    *,
    dtype: Any = None,
    backend: BackendLike = None,
) -> tuple[Any, Any]:
    """``(y, x)`` coordinate grids for a ``(ny, nx)`` array (CONVENTIONS 1.1-1.2).

    ``x`` varies along axis 1 (columns) and ``y`` along axis 0 (rows). Both
    are broadcast views of 1-D coordinate vectors (read-only on NumPy; copy
    them before writing). ``dtype`` and
    ``backend`` are as in :func:`centered_coordinates` (default: host
    float64).
    """
    ny, nx = (int(shape), int(shape)) if np.ndim(shape) == 0 else (int(shape[0]), int(shape[1]))
    xp = get_backend(backend).xp
    y = centered_coordinates(ny, pitch, dtype=dtype, backend=backend)[:, None]
    x = centered_coordinates(nx, pitch, dtype=dtype, backend=backend)[None, :]
    return xp.broadcast_to(y, (ny, nx)), xp.broadcast_to(x, (ny, nx))


def centroid(image: Any) -> tuple[float, float]:
    """Intensity centroid ``(y, x)`` in the pixel-centre coordinates of 1.2 (pixels)."""
    data = np.asarray(image, dtype=np.float64)
    total = data.sum()
    if not total > 0:
        raise ValueError("image has no positive flux")
    y, x = coordinate_grid(data.shape)
    return float((data * y).sum() / total), float((data * x).sum() / total)


def opd_to_phase(opd: Any, wavelength: float) -> Any:
    """Phase in radians of an OPD in metres at ``wavelength`` (CONVENTIONS 2.2)."""
    return np.multiply(opd, 2.0 * math.pi / wavelength)


def phase_to_opd(phase: Any, wavelength: float) -> Any:
    """OPD in metres of a phase in radians at ``wavelength`` (CONVENTIONS 2.2)."""
    return np.multiply(phase, wavelength / (2.0 * math.pi))


def rms(opd: Any, amplitude: Any) -> float:
    """Intensity-weighted, piston-removed RMS over a pupil (CONVENTIONS 4.1).

    ``amplitude`` is the pupil amplitude; pixels where it is zero do not count.
    """
    data = np.asarray(opd, dtype=np.float64)
    weight = np.asarray(amplitude, dtype=np.float64) ** 2
    if data.shape != weight.shape:
        raise ValueError(f"opd {data.shape} and amplitude {weight.shape} differ in shape")
    total = weight.sum()
    if not total > 0:
        raise ValueError("pupil has no transmission")
    mean = (weight * data).sum() / total
    return float(math.sqrt((weight * (data - mean) ** 2).sum() / total))


def strehl_marechal(rms_opd: float, wavelength: float) -> float:
    """Maréchal approximation ``exp(-(2 pi rms / lambda)^2)`` (CONVENTIONS 4.2)."""
    return float(math.exp(-((2.0 * math.pi * rms_opd / wavelength) ** 2)))
