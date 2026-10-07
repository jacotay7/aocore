"""Executable checks of the stack conventions (CONVENTIONS.md).

Every package runs the checks that apply to it in its own test suite, against
its own implementation, for example::

    from aocore import conformance

    def test_tilt_moves_the_image_towards_positive_x():
        conformance.check_tilt_direction(my_image_from_opd, pupil_shape=(64, 64),
                                         pitch=8.0 / 64, wavelength=1.6e-6,
                                         pixel_scale=1.6e-6 / 8.0 / 2)

Each ``check_*`` function raises :class:`ConformanceError` (a subclass of
``AssertionError``, so pytest reports it as a failure) with the rule number
and the measured deviation, and returns a short dict of what it measured.
The checks take plain callables and NumPy arrays, so they cannot depend on
the internals of the package under test.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .conventions import centroid, coordinate_grid, rms

__all__ = [
    "ConformanceError",
    "check_coordinates",
    "check_image_centring",
    "check_rms",
    "check_slope_sign",
    "check_tilt_direction",
    "check_unit_flux",
    "check_wind_motion",
    "check_zernike_basis",
]


class ConformanceError(AssertionError):
    """A convention from CONVENTIONS.md is violated."""


def _fail(rule: str, message: str) -> None:
    raise ConformanceError(f"CONVENTIONS {rule}: {message}")


def _as_host(value: Any) -> np.ndarray:
    try:
        import cupy

        if isinstance(value, cupy.ndarray):
            return np.asarray(cupy.asnumpy(value))
    except ImportError:  # pragma: no cover - CuPy optional
        pass
    return np.asarray(value)


def _circular_pupil(shape: tuple[int, int]) -> np.ndarray:
    y, x = coordinate_grid(shape)
    radius = min(shape) / 2.0
    return (np.hypot(x, y) <= radius).astype(np.float64)


# ------------------------------------------------------------------ section 1
def check_coordinates(
    coordinates: Callable[[int, float], Any], *, tol: float = 1e-12
) -> dict[str, float]:
    """Rule 1.2: ``coordinates(n, pitch)`` returns ``(i - (n - 1) / 2) * pitch``.

    Checked for an even and an odd ``n``.
    """
    worst = 0.0
    for n in (6, 7):
        got = _as_host(coordinates(n, 0.5)).astype(np.float64).ravel()
        expected = (np.arange(n) - (n - 1) / 2.0) * 0.5
        if got.shape != expected.shape:
            _fail("1.2", f"coordinates({n}, 0.5) has shape {got.shape}, expected {expected.shape}")
        worst = max(worst, float(np.max(np.abs(got - expected))))
    if worst > tol:
        _fail("1.2", f"pixel centres deviate from (i - (n-1)/2) * pitch by {worst:.3g}")
    return {"max_deviation": worst}


def check_image_centring(
    image_from_opd: Callable[[np.ndarray], Any],
    *,
    pupil_shape: tuple[int, int],
    tol_pixels: float = 0.02,
) -> dict[str, float]:
    """Rule 1.3: a flat wavefront images to the window centre ``(n - 1) / 2``.

    ``image_from_opd(opd)`` returns one 2-D image for a ``pupil_shape`` OPD map
    in metres (the package's own pupil, sampling and window). The image must be
    centro-symmetric about the optical axis, so its centroid measures where the
    axis lies.
    """
    image = _as_host(image_from_opd(np.zeros(pupil_shape)))
    cy, cx = centroid(image)
    if max(abs(cy), abs(cx)) > tol_pixels:
        _fail(
            "1.3",
            f"the flat-wavefront image is centred at ({cy:+.3f}, {cx:+.3f}) pixels from "
            "(n-1)/2; the fftshift convention (axis on pixel n/2) gives (+0.5, +0.5)",
        )
    return {"centroid_y": cy, "centroid_x": cx}


# ------------------------------------------------------------------ section 3
def check_tilt_direction(
    image_from_opd: Callable[[np.ndarray], Any],
    *,
    pupil_shape: tuple[int, int],
    pitch: float,
    pixel_scale: float,
    pixels: float = 3.0,
    rel_tol: float = 0.1,
) -> dict[str, float]:
    """Rules 1.1 and 3.1: an OPD ramp ``a x`` moves the image by ``a`` radians towards +x.

    Applies ramps of ``pixels`` detector pixels along x and then along y, and
    checks the centroid moves by that amount along columns, and then rows, in
    the positive direction.

    Parameters
    ----------
    image_from_opd:
        Maps an OPD map (metres, ``pupil_shape``) to one image.
    pupil_shape:
        Shape of the OPD maps the callable accepts.
    pitch:
        Pupil pixel pitch in metres.
    pixel_scale:
        Detector pixel scale in radians.
    """
    y, x = coordinate_grid(pupil_shape, pitch)
    flat = centroid(_as_host(image_from_opd(np.zeros(pupil_shape))))
    slope = pixels * pixel_scale
    tilted_x = centroid(_as_host(image_from_opd(slope * x)))
    tilted_y = centroid(_as_host(image_from_opd(slope * y)))
    shift_x = (tilted_x[0] - flat[0], tilted_x[1] - flat[1])
    shift_y = (tilted_y[0] - flat[0], tilted_y[1] - flat[1])
    if not (abs(shift_x[1] - pixels) <= rel_tol * pixels and abs(shift_x[0]) <= rel_tol * pixels):
        _fail(
            "3.1",
            f"an OPD ramp along +x of {pixels} pixels moved the image by (dy, dx) = "
            f"({shift_x[0]:+.2f}, {shift_x[1]:+.2f}) pixels; expected (0, +{pixels})",
        )
    if not (abs(shift_y[0] - pixels) <= rel_tol * pixels and abs(shift_y[1]) <= rel_tol * pixels):
        _fail(
            "3.1",
            f"an OPD ramp along +y of {pixels} pixels moved the image by (dy, dx) = "
            f"({shift_y[0]:+.2f}, {shift_y[1]:+.2f}) pixels; expected (+{pixels}, 0)",
        )
    return {"shift_x_pixels": shift_x[1], "shift_y_pixels": shift_y[0]}


def check_unit_flux(
    psf_from_opd: Callable[[np.ndarray], Any],
    *,
    pupil_shape: tuple[int, int],
    rel_tol: float = 1e-3,
) -> dict[str, float]:
    """Rule 3.3: a normalized image sums to 1 when the window captures all the light.

    The callable must return an image on a window that holds essentially all
    of the light (for an FFT model, the whole padded grid).
    """
    total = float(_as_host(psf_from_opd(np.zeros(pupil_shape))).sum())
    if abs(total - 1.0) > rel_tol:
        _fail("3.3", f"the normalized image sums to {total:.6f}, not 1")
    return {"total": total}


# ------------------------------------------------------------------ section 4
def check_rms(
    rms_fn: Callable[[np.ndarray, np.ndarray], float], *, rel_tol: float = 1e-9
) -> dict[str, float]:
    """Rule 4.1: ``rms_fn(opd, amplitude)`` is intensity-weighted with piston removed.

    Uses a grey-edged pupil and an OPD with a large piston, so an unweighted
    or piston-keeping definition fails.
    """
    rng = np.random.default_rng(0)
    amplitude = _circular_pupil((32, 32))
    amplitude[amplitude > 0] *= rng.uniform(0.2, 1.0, int(amplitude.sum()))
    opd = (rng.standard_normal((32, 32)) * 1e-8 + 5e-7) * (amplitude > 0)
    expected = rms(opd, amplitude)
    got = float(rms_fn(opd, amplitude))
    if abs(got - expected) > rel_tol * expected:
        _fail(
            "4.1",
            f"rms is {got:.6g} m; the intensity-weighted piston-removed value is {expected:.6g} m",
        )
    return {"rms": got}


# ------------------------------------------------------------------ section 5
def check_zernike_basis(
    zernike: Callable[[int, np.ndarray, np.ndarray], Any],
    *,
    n: int = 128,
    tol: float = 2e-2,
) -> dict[str, float]:
    """Rules 5.1 and 5.2: Noll ordering, unit RMS over the disc, tip along +x.

    ``zernike(j, y, x)`` evaluates Noll mode ``j`` at unit-disc coordinates
    ``(y, x)`` (1-D arrays of points inside the disc). Checks unit RMS for
    j = 2..11, tip ``= 2 rho cos(theta)`` along +x, tilt along +y, and defocus
    ``= sqrt(3) (2 rho^2 - 1)``.
    """
    y, x = coordinate_grid((n, n), 2.0 / n)
    inside = np.hypot(x, y) <= 1.0
    py, px = y[inside], x[inside]
    rho, theta = np.hypot(px, py), np.arctan2(py, px)
    worst = 0.0
    for j in range(2, 12):
        mode = _as_host(zernike(j, py, px)).astype(np.float64).ravel()
        worst = max(worst, abs(float(np.sqrt(np.mean(mode**2))) - 1.0))
    if worst > tol:
        _fail("5.1", f"Noll modes 2-11 deviate from unit RMS by up to {worst:.3g}")
    expected = {
        2: 2 * rho * np.cos(theta),
        3: 2 * rho * np.sin(theta),
        4: math.sqrt(3) * (2 * rho**2 - 1),
    }
    for j, ref in expected.items():
        mode = _as_host(zernike(j, py, px)).astype(np.float64).ravel()
        err = float(np.max(np.abs(mode - ref)))
        if err > 1e-6:
            rule = "5.2" if j in (2, 3) else "5.1"
            _fail(rule, f"Noll j={j} differs from its definition by {err:.3g} (axis swap or sign?)")
    return {"max_rms_deviation": worst}


# ------------------------------------------------------------------ section 6
def check_wind_motion(
    frames: Callable[[float], tuple[Any, Any, float]],
    *,
    tol_pixels: float = 0.5,
) -> dict[str, float]:
    """Rule 6.1: the frozen-flow pattern moves along ``-wind_vector``.

    ``frames(direction_deg)`` builds a single frozen-flow layer blowing from
    ``direction_deg`` and returns ``(frame_0, frame_1, displacement_pixels)``:
    two OPD maps one time step apart and the expected displacement magnitude in
    pixels (wind speed x dt / pitch). Checked for winds from +x (0 degrees) and
    from +y (90 degrees) by cross-correlation.
    """

    def shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
        corr = np.real(np.fft.ifft2(np.conj(fa) * fb))
        iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
        ny, nx = corr.shape
        return float((iy + ny // 2) % ny - ny // 2), float((ix + nx // 2) % nx - nx // 2)

    out = {}
    for direction, expect in ((0.0, (0.0, -1.0)), (90.0, (-1.0, 0.0))):
        a, b, pixels = frames(direction)
        dy, dx = shift(_as_host(a).astype(np.float64), _as_host(b).astype(np.float64))
        ey, ex = expect[0] * pixels, expect[1] * pixels
        if abs(dy - ey) > tol_pixels or abs(dx - ex) > tol_pixels:
            _fail(
                "6.1",
                f"wind from {direction:g} deg moved the pattern by (dy, dx) = "
                f"({dy:+.1f}, {dx:+.1f}) pixels; expected ({ey:+.1f}, {ex:+.1f}) "
                "(along -wind_vector, x = columns)",
            )
        out[f"shift_{int(direction)}"] = dx if direction == 0 else dy
    return out


# ------------------------------------------------------------------ section 7
def check_slope_sign(
    slopes_from_opd: Callable[[np.ndarray], Any],
    *,
    pupil_shape: tuple[int, int],
    pitch: float,
    gradient: float,
    n_subapertures: int,
) -> dict[str, float]:
    """Rules 7.1 and 7.2: x-slopes first, positive for a positive OPD gradient along x.

    ``slopes_from_opd(opd)`` returns the slope vector ``[sx..., sy...]`` (length
    ``2 * n_subapertures``) for an OPD map. A ramp ``gradient * x`` must give
    positive mean x-slopes and near-zero y-slopes, and a ramp along y the
    reverse.
    """
    y, x = coordinate_grid(pupil_shape, pitch)
    sx_ramp = _as_host(slopes_from_opd(gradient * x)).astype(np.float64).ravel()
    sy_ramp = _as_host(slopes_from_opd(gradient * y)).astype(np.float64).ravel()
    if sx_ramp.size != 2 * n_subapertures:
        _fail("7.1", f"slope vector has {sx_ramp.size} entries, expected 2 x {n_subapertures}")
    half = n_subapertures
    mx, my = sx_ramp[:half].mean(), sx_ramp[half:].mean()
    if not (mx > 0 and abs(my) < 0.1 * abs(mx)):
        _fail("7.2", f"an OPD ramp along +x gave mean slopes (x, y) = ({mx:.3g}, {my:.3g})")
    nx_, ny_ = sy_ramp[:half].mean(), sy_ramp[half:].mean()
    if not (ny_ > 0 and abs(nx_) < 0.1 * abs(ny_)):
        _fail("7.2", f"an OPD ramp along +y gave mean slopes (x, y) = ({nx_:.3g}, {ny_:.3g})")
    return {"x_ramp_mean_sx": mx, "y_ramp_mean_sy": ny_}
