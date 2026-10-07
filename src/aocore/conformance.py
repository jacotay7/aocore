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

Packages whose images come from an OPD (Fourier optics) use the
``image_from_opd`` checks. Packages that render analytic point-source images
(Gaussian, Moffat, Airy profiles) use the ``render`` variants:
:func:`check_point_source_centring`, :func:`check_point_source_flux` and
:func:`check_edge_flux_loss`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .conventions import centroid, coordinate_grid, rms

__all__ = [
    "ConformanceError",
    "check_coordinates",
    "check_edge_flux_loss",
    "check_image_centring",
    "check_point_source_centring",
    "check_point_source_flux",
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


def _shape2(shape: Any) -> tuple[int, int]:
    return int(shape[0]), int(shape[1])


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
    return _check_centred(_as_host(image_from_opd(np.zeros(pupil_shape))), tol_pixels)


def _check_centred(image: np.ndarray, tol_pixels: float) -> dict[str, float]:
    cy, cx = centroid(image)
    if max(abs(cy), abs(cx)) > tol_pixels:
        _fail(
            "1.3",
            f"the on-axis image of shape {image.shape} is centred at ({cy:+.3f}, {cx:+.3f}) "
            "pixels from (n-1)/2; the fftshift convention (axis on pixel n/2) gives "
            "(+0.5, +0.5)",
        )
    return {"centroid_y": cy, "centroid_x": cx}


def check_point_source_centring(
    render: Callable[[tuple[int, int]], Any],
    *,
    shapes: Sequence[tuple[int, int]] = ((32, 32), (33, 33)),
    tol_pixels: float = 0.02,
) -> dict[str, float]:
    """Rule 1.3 for analytic images: an on-axis point source lands on ``(n - 1) / 2``.

    The image-builder form of :func:`check_image_centring`, for packages whose
    images are rendered from a PSF model rather than propagated from an OPD.
    ``render(shape)`` returns the noise-free image, of ``shape`` pixels, of a
    point source on the optical axis (zero field offset). The PSF must be
    centro-symmetric (Gaussian, Moffat, Airy, ...), so its centroid measures
    where the axis lies.

    Checked for every window in ``shapes``. Keep an even size among them: for
    odd ``n`` the ``fftshift`` convention (axis on pixel ``n // 2``) coincides
    with ``(n - 1) / 2`` and cannot be told apart. Returns the worst centroid.
    """
    worst = {"centroid_y": 0.0, "centroid_x": 0.0}
    for shape in shapes:
        found = _check_centred(_as_host(render(_shape2(shape))), tol_pixels)
        if max(map(abs, found.values())) >= max(map(abs, worst.values())):
            worst = found
    return worst


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
    return _check_total(_as_host(psf_from_opd(np.zeros(pupil_shape))), 1.0, rel_tol)


def _check_total(image: np.ndarray, flux: float, rel_tol: float) -> dict[str, float]:
    total = float(image.astype(np.float64).sum())
    if abs(total - flux) > rel_tol * abs(flux):
        _fail("3.3", f"the normalized image sums to {total:.6f}, not {flux:g}")
    return {"total": total}


def check_point_source_flux(
    render: Callable[[tuple[int, int]], Any],
    *,
    shape: tuple[int, int] = (128, 128),
    flux: float = 1.0,
    rel_tol: float = 1e-3,
) -> dict[str, float]:
    """Rule 3.3 for analytic images: a point source of unit flux sums to ``flux``.

    The image-builder form of :func:`check_unit_flux`. ``render(shape)``
    returns the noise-free image of an on-axis point source whose total flux
    is ``flux`` (1 for a normalized PSF). ``shape`` must be large enough to
    hold essentially all the light (to ``rel_tol``); for heavy-winged
    profiles such as a Moffat with small beta, pick a larger window.
    """
    return _check_total(_as_host(render(_shape2(shape))), flux, rel_tol)


def check_edge_flux_loss(
    render: Callable[[tuple[int, int], tuple[float, float]], Any],
    *,
    shape: tuple[int, int] = (32, 32),
    pad: int = 32,
    rel_tol: float = 1e-3,
    half_tol: float = 0.05,
    symmetric: bool = True,
) -> dict[str, float]:
    """Rule 3.3: a window clips a source at its edge and does not renormalize.

    ``render(shape, position)`` returns the noise-free image, of ``shape``
    pixels, of one point source at ``position = (y, x)`` in pixels from the
    window centre (the coordinates of rule 1.2, as :func:`aocore.centroid`
    reports them: pixel ``i`` is at ``i - (n - 1) / 2``). The source's total
    flux must not depend on the window.

    For a source centred on each of the four window edges (``x = +-nx/2``,
    ``y = +-ny/2``), the check requires

    * that the window holds exactly the visible part of the source: the same
      source rendered in a window larger by ``pad`` pixels on every side,
      cropped back to ``shape``, has the same total flux (to ``rel_tol``);
    * with ``symmetric`` (a centro-symmetric PSF narrow compared with the
      window), that the edge source deposits about half (to ``half_tol``) the
      flux of a source in the middle of the window.

    A model that renormalizes the clipped stamp to its full flux deposits as
    much at the edge as in the middle, and fails both.
    """
    if pad < 1:
        raise ValueError("pad must be >= 1")
    ny, nx = _shape2(shape)
    big = (ny + 2 * pad, nx + 2 * pad)
    middle = float(_as_host(render((ny, nx), (0.0, 0.0))).astype(np.float64).sum())
    if not middle > 0:
        _fail("3.3", "a source in the middle of the window deposits no flux")
    out: dict[str, float] = {"middle": middle}
    worst_ratio = 0.5
    edges = {"+x": (0.0, nx / 2.0), "-x": (0.0, -nx / 2.0), "+y": (ny / 2.0, 0.0)}
    edges["-y"] = (-ny / 2.0, 0.0)
    for name, position in edges.items():
        edge = float(_as_host(render((ny, nx), position)).astype(np.float64).sum())
        large = _as_host(render(big, position)).astype(np.float64)
        visible = float(large[pad : pad + ny, pad : pad + nx].sum())
        if abs(edge - visible) > rel_tol * max(abs(visible), 1e-300):
            _fail(
                "3.3",
                f"a source on the {name} edge deposits {edge:.6g} in the window, but only "
                f"{visible:.6g} of it falls there (rendered in a larger window); the clipped "
                "light must be lost, not renormalized away",
            )
        ratio = edge / middle
        if symmetric and abs(ratio - 0.5) > half_tol:
            _fail(
                "3.3",
                f"a source on the {name} edge deposits {ratio:.3f} of the flux of a source "
                "in the middle; a symmetric PSF clipped at its centre keeps about half",
            )
        out[f"edge_{name}"] = edge
        if abs(ratio - 0.5) >= abs(worst_ratio - 0.5):
            worst_ratio = ratio
    out["worst_ratio"] = worst_ratio
    return out


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
    pixels (wind speed x dt / pitch, a few pixels, and well below the frame
    size). Checked for winds from +x (0 degrees) and from +y (90 degrees) by a
    least-squares search of integer shifts over the frames' interior.
    """

    def shift(a: np.ndarray, b: np.ndarray, reach: int) -> tuple[float, float]:
        # Least-squares integer shift over the frames' interior. Turbulence is
        # red and the frames are not periodic, so an FFT cross-correlation peak
        # is dominated by the largest scales and can sit at zero shift.
        margin = reach + 1
        target = b[margin:-margin, margin:-margin]
        best = (np.inf, 0, 0)
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                moved = np.roll(a, (dy, dx), axis=(0, 1))[margin:-margin, margin:-margin]
                err = float(np.mean((moved - target) ** 2))
                if err < best[0]:
                    best = (err, dy, dx)
        return float(best[1]), float(best[2])

    out = {}
    for direction, expect in ((0.0, (0.0, -1.0)), (90.0, (-1.0, 0.0))):
        a, b, pixels = frames(direction)
        reach = int(np.ceil(pixels)) + 2
        dy, dx = shift(_as_host(a).astype(np.float64), _as_host(b).astype(np.float64), reach)
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
    layout: str = "blocked",
) -> dict[str, float]:
    """Rules 7.1 and 7.2: the declared slope layout, positive for a positive OPD gradient.

    ``slopes_from_opd(opd)`` returns the slope vector (length
    ``2 * n_subapertures``) for an OPD map, in the declared ``layout``:
    ``"blocked"`` (``[sx..., sy...]``) or ``"interleaved"``
    (``[sx_1, sy_1, ...]``). A ramp ``gradient * x`` must give positive mean
    x-slopes and near-zero y-slopes, and a ramp along y the reverse, so a
    swapped layout or a flipped sign fails.
    """
    if layout not in ("blocked", "interleaved"):
        raise ValueError("layout must be 'blocked' or 'interleaved'")
    y, x = coordinate_grid(pupil_shape, pitch)
    sx_ramp = _as_host(slopes_from_opd(gradient * x)).astype(np.float64).ravel()
    sy_ramp = _as_host(slopes_from_opd(gradient * y)).astype(np.float64).ravel()
    if sx_ramp.size != 2 * n_subapertures:
        _fail("7.1", f"slope vector has {sx_ramp.size} entries, expected 2 x {n_subapertures}")

    def split(vector: np.ndarray) -> tuple[float, float]:
        if layout == "interleaved":
            return float(vector[0::2].mean()), float(vector[1::2].mean())
        return float(vector[:n_subapertures].mean()), float(vector[n_subapertures:].mean())

    mx, my = split(sx_ramp)
    if not (mx > 0 and abs(my) < 0.1 * abs(mx)):
        _fail("7.2", f"an OPD ramp along +x gave mean slopes (x, y) = ({mx:.3g}, {my:.3g})")
    nx_, ny_ = split(sy_ramp)
    if not (ny_ > 0 and abs(nx_) < 0.1 * abs(ny_)):
        _fail("7.2", f"an OPD ramp along +y gave mean slopes (x, y) = ({nx_:.3g}, {ny_:.3g})")
    return {"x_ramp_mean_sx": mx, "y_ramp_mean_sy": ny_}
