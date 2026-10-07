"""Reference implementations of the stack conventions (see CONVENTIONS.md).

Small, dependency-free definitions that the conformance checks compare
against, and that packages can import instead of re-deriving.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

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


def centered_coordinates(n: int, pitch: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Pixel-centre coordinates ``(i - (n - 1) / 2 + offset) * pitch`` (CONVENTIONS 1.2)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0 + offset) * pitch


def coordinate_grid(shape: Any, pitch: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(y, x)`` coordinate grids for a ``(ny, nx)`` array (CONVENTIONS 1.1-1.2).

    ``x`` varies along axis 1 (columns) and ``y`` along axis 0 (rows).
    """
    ny, nx = (int(shape), int(shape)) if np.ndim(shape) == 0 else (int(shape[0]), int(shape[1]))
    y = centered_coordinates(ny, pitch)[:, None]
    x = centered_coordinates(nx, pitch)[None, :]
    return np.broadcast_to(y, (ny, nx)), np.broadcast_to(x, (ny, nx))


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
