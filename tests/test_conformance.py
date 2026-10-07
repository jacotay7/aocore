"""Each check passes a conforming implementation and catches violations seen in the stack."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aocore import conformance as cf
from aocore.conventions import centered_coordinates, coordinate_grid, rms

N, Q, LAM = 32, 4, 1e-6
PITCH = 1.0 / N
PIXEL_SCALE = LAM / (N * PITCH) / Q


def _pupil() -> np.ndarray:
    y, x = coordinate_grid((N, N))
    return (np.hypot(x, y) < N / 2).astype(float)


def _image(
    opd: np.ndarray, *, centring: str = "between", transpose: bool = False, sign: int = -1
) -> np.ndarray:
    """Fraunhofer image on a Q-times padded grid, cropped to N x N."""
    big = N * Q
    field = np.zeros((big, big), complex)
    u = _pupil() * np.exp(2j * np.pi * (opd.T if transpose else opd) / LAM)
    field[:N, :N] = u
    # The pixel-centre convention puts the axis at (n-1)/2: a half-pixel input ramp.
    if centring == "between":
        i = np.arange(N)
        ramp = np.exp(-1j * np.pi * (i[:, None] + i[None, :]) / big)
        field[:N, :N] *= ramp
    spec = np.fft.fft2(field, norm="ortho") if sign < 0 else np.fft.ifft2(field, norm="ortho")
    img = np.abs(np.fft.fftshift(spec)) ** 2 / _pupil().sum()
    start = (big - N) // 2
    return img[start : start + N, start : start + N]


def test_coordinates() -> None:
    cf.check_coordinates(centered_coordinates)
    with pytest.raises(cf.ConformanceError, match=r"1\.2"):
        cf.check_coordinates(lambda n, pitch: (np.arange(n) - n / 2) * pitch)


def test_image_centring_catches_the_fftshift_convention() -> None:
    cf.check_image_centring(_image, pupil_shape=(N, N))
    with pytest.raises(cf.ConformanceError, match=r"1\.3"):
        cf.check_image_centring(lambda o: _image(o, centring="fftshift"), pupil_shape=(N, N))


def test_tilt_direction_catches_transposed_axes_and_flipped_sign() -> None:
    kw = {"pupil_shape": (N, N), "pitch": PITCH, "pixel_scale": PIXEL_SCALE}
    cf.check_tilt_direction(_image, **kw)
    with pytest.raises(cf.ConformanceError, match=r"3\.1"):
        cf.check_tilt_direction(lambda o: _image(o, transpose=True), **kw)
    with pytest.raises(cf.ConformanceError, match=r"3\.1"):
        cf.check_tilt_direction(lambda o: _image(o, sign=+1), **kw)


def test_unit_flux() -> None:
    def full(opd: np.ndarray) -> np.ndarray:
        field = np.zeros((N * Q, N * Q), complex)
        field[:N, :N] = _pupil() * np.exp(2j * np.pi * opd / LAM)
        return np.abs(np.fft.fft2(field, norm="ortho")) ** 2 / _pupil().sum()

    cf.check_unit_flux(full, pupil_shape=(N, N))
    with pytest.raises(cf.ConformanceError, match=r"3\.3"):
        cf.check_unit_flux(lambda o: 2 * full(o), pupil_shape=(N, N))


def test_rms_catches_unweighted_and_piston_keeping_definitions() -> None:
    cf.check_rms(rms)
    with pytest.raises(cf.ConformanceError, match=r"4\.1"):
        cf.check_rms(lambda opd, amp: float(np.sqrt(np.mean(opd[amp > 0] ** 2))))
    with pytest.raises(cf.ConformanceError, match=r"4\.1"):
        cf.check_rms(lambda opd, amp: float(np.std(opd[amp > 0])))


def _noll(j: int) -> tuple[int, int]:
    n, j1 = 0, j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + ((n + 1) % 2)) // 2))
    return n, m


def _zernike(j: int, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    n, m = _noll(j)
    rho, theta = np.hypot(x, y), np.arctan2(y, x)
    radial = sum(
        (-1) ** k
        * math.factorial(n - k)
        / (
            math.factorial(k)
            * math.factorial((n + abs(m)) // 2 - k)
            * math.factorial((n - abs(m)) // 2 - k)
        )
        * rho ** (n - 2 * k)
        for k in range((n - abs(m)) // 2 + 1)
    )
    norm = math.sqrt(n + 1) * (1 if m == 0 else math.sqrt(2))
    return norm * radial * (np.cos(m * theta) if m >= 0 else np.sin(-m * theta))


def test_zernike_basis_catches_transposition_and_double_normalisation() -> None:
    cf.check_zernike_basis(_zernike)
    with pytest.raises(cf.ConformanceError, match=r"5\.2"):
        cf.check_zernike_basis(lambda j, y, x: _zernike(j, x, y))  # pyturb 1.x frame
    with pytest.raises(cf.ConformanceError, match=r"5\.1"):
        cf.check_zernike_basis(lambda j, y, x: _zernike(j, y, x) * math.sqrt(_noll(j)[0] + 1))


def test_wind_motion() -> None:
    # Red, non-periodic screens like real turbulence (2-D random walks), so an
    # estimator dominated by the largest scales would miss the shift.
    rng_big = np.random.default_rng(2)

    def frames(direction: float, axis_zero_is_x: bool = False):
        theta = math.radians(direction)
        vx, vy = 3 * math.cos(theta), 3 * math.sin(theta)
        if axis_zero_is_x:
            vx, vy = vy, vx
        # phi(x, t) = phi_0(x + v t): the pattern moves along -v.
        big = np.cumsum(np.cumsum(rng_big.standard_normal((80, 80)), axis=0), axis=1)
        first = big[8:72, 8:72]
        moved = big[8 + round(vy) : 72 + round(vy), 8 + round(vx) : 72 + round(vx)]
        return first, moved, 3.0

    cf.check_wind_motion(frames)
    with pytest.raises(cf.ConformanceError, match=r"6\.1"):
        cf.check_wind_motion(lambda d: frames(d, axis_zero_is_x=True))


def test_slope_sign() -> None:
    def slopes(opd: np.ndarray, swap: bool = False, interleave: bool = False) -> np.ndarray:
        gy, gx = np.gradient(opd)
        sub = gx.reshape(4, 8, 4, 8).mean(axis=(1, 3)).ravel()
        suby = gy.reshape(4, 8, 4, 8).mean(axis=(1, 3)).ravel()
        if interleave:
            return np.stack([sub, suby], axis=1).ravel()
        return np.concatenate([suby, sub] if swap else [sub, suby])

    kw = {"pupil_shape": (N, N), "pitch": PITCH, "gradient": 1e-6, "n_subapertures": 16}
    cf.check_slope_sign(slopes, **kw)
    cf.check_slope_sign(lambda o: slopes(o, interleave=True), layout="interleaved", **kw)
    with pytest.raises(cf.ConformanceError, match="7"):
        cf.check_slope_sign(lambda o: slopes(o, swap=True), **kw)
    with pytest.raises(cf.ConformanceError, match="7"):
        cf.check_slope_sign(lambda o: slopes(o, interleave=True), **kw)
    with pytest.raises(ValueError, match="layout"):
        cf.check_slope_sign(slopes, layout="zigzag", **kw)


# ------------------------------------------------- analytic point-source images
def _erf_axis(n: int, centre: float, sigma: float) -> np.ndarray:
    """Pixel-integrated 1-D Gaussian; pixel i spans i - (n-1)/2 -+ 1/2 (rule 1.2)."""
    from scipy.special import erf

    edges = np.arange(n + 1) - n / 2.0
    return np.diff(0.5 * (1.0 + erf((edges - centre) / (sigma * math.sqrt(2.0)))))


def _gaussian(
    shape: tuple[int, int],
    position: tuple[float, float] = (0.0, 0.0),
    *,
    sigma: float = 1.5,
    renormalize: bool = False,
    fftshift: bool = False,
) -> np.ndarray:
    """Unit-flux Gaussian, integrated over each pixel and clipped by the window."""
    ny, nx = shape
    cy, cx = position
    if fftshift:  # axis on pixel n // 2 instead of (n - 1) / 2
        cy, cx = cy + ny // 2 - (ny - 1) / 2, cx + nx // 2 - (nx - 1) / 2
    image = np.outer(_erf_axis(ny, cy, sigma), _erf_axis(nx, cx, sigma))
    return image / image.sum() if renormalize else image


def _moffat(
    shape: tuple[int, int],
    position: tuple[float, float] = (0.0, 0.0),
    *,
    alpha: float = 2.0,
    beta: float = 3.0,
    renormalize: bool = False,
    periodic: bool = False,
) -> np.ndarray:
    """Unit-flux Moffat sampled at pixel centres, as getframes renders it."""
    y, x = coordinate_grid(shape)
    dy, dx = y - position[0], x - position[1]
    if periodic:  # an FFT model without padding wraps the light around
        dy = (dy + shape[0] / 2) % shape[0] - shape[0] / 2
        dx = (dx + shape[1] / 2) % shape[1] - shape[1] / 2
    image = (beta - 1) / (math.pi * alpha**2) * (1 + (dx**2 + dy**2) / alpha**2) ** -beta
    return image / image.sum() if renormalize else image


def test_point_source_centring_catches_the_fftshift_convention() -> None:
    found = cf.check_point_source_centring(lambda s: _gaussian(s))
    assert abs(found["centroid_x"]) < 1e-9
    cf.check_point_source_centring(lambda s: _moffat(s), shapes=[(48, 40)])
    with pytest.raises(cf.ConformanceError, match=r"1\.3"):
        cf.check_point_source_centring(lambda s: _gaussian(s, fftshift=True))
    # Odd windows cannot tell the two conventions apart (documented).
    cf.check_point_source_centring(lambda s: _gaussian(s, fftshift=True), shapes=[(33, 33)])


def test_point_source_flux() -> None:
    assert cf.check_point_source_flux(lambda s: _gaussian(s))["total"] == pytest.approx(1.0)
    cf.check_point_source_flux(lambda s: 250.0 * _gaussian(s), flux=250.0)
    with pytest.raises(cf.ConformanceError, match=r"3\.3"):
        cf.check_point_source_flux(lambda s: 2 * _gaussian(s))
    # A window too small for the PSF loses light: the check says so.
    with pytest.raises(cf.ConformanceError, match=r"3\.3"):
        cf.check_point_source_flux(lambda s: _gaussian(s, sigma=4.0), shape=(16, 16))


def test_edge_flux_loss_passes_clipping_renderers() -> None:
    found = cf.check_edge_flux_loss(lambda s, p: _gaussian(s, p))
    assert found["worst_ratio"] == pytest.approx(0.5, abs=1e-6)
    found = cf.check_edge_flux_loss(lambda s, p: _moffat(s, p), shape=(40, 48))
    assert found["worst_ratio"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize(
    "render",
    [
        lambda s, p: _gaussian(s, p, renormalize=True),
        lambda s, p: _moffat(s, p, renormalize=True),
        lambda s, p: _moffat(s, p, periodic=True),
    ],
    ids=["gaussian-renormalized", "moffat-renormalized", "moffat-wrapped"],
)
def test_edge_flux_loss_catches_renormalized_stamps(render) -> None:
    with pytest.raises(cf.ConformanceError, match=r"3\.3"):
        cf.check_edge_flux_loss(render)
    # The comparison with a larger window catches it without the symmetry test.
    with pytest.raises(cf.ConformanceError, match="renormalized away"):
        cf.check_edge_flux_loss(render, symmetric=False)


def test_edge_flux_loss_asymmetric_psf_and_errors() -> None:
    def lopsided(shape, position, renormalize=False):
        # Two blobs: the source's light is not symmetric about its position.
        main = _gaussian(shape, position)
        side = _gaussian(shape, (position[0] + 2.0, position[1] + 3.0))
        image = 0.7 * main + 0.3 * side
        return image / image.sum() if renormalize else image

    with pytest.raises(cf.ConformanceError, match="about half"):
        cf.check_edge_flux_loss(lopsided)
    cf.check_edge_flux_loss(lopsided, symmetric=False)
    with pytest.raises(cf.ConformanceError, match="renormalized away"):
        cf.check_edge_flux_loss(lambda s, p: lopsided(s, p, True), symmetric=False)
    with pytest.raises(cf.ConformanceError, match="no flux"):
        cf.check_edge_flux_loss(lambda s, p: np.zeros(s))
    with pytest.raises(ValueError, match="pad"):
        cf.check_edge_flux_loss(lambda s, p: _gaussian(s, p), pad=0)
