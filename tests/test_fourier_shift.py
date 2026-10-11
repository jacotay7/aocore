"""aocore.fourier_shift: sign (CONVENTIONS 1.1, 3.1), exactness, adjoint, devices."""

from __future__ import annotations

import numpy as np
import pytest

import aocore as ac
from conftest import devices


def test_whole_pixel_shifts_equal_a_roll(rng) -> None:
    a = rng.standard_normal((3, 16, 17))
    rolled = np.roll(a, (-3, 2), (-2, -1))
    np.testing.assert_allclose(ac.fourier_shift(a, 2, -3), rolled, atol=1e-13)
    z = a + 1j * rng.standard_normal(a.shape)
    out = ac.fourier_shift(z, -5, 4)
    assert np.iscomplexobj(out)
    np.testing.assert_allclose(out, np.roll(z, (4, -5), (-2, -1)), atol=1e-13)


def test_positive_dx_moves_content_towards_larger_column_index() -> None:
    y, x = ac.coordinate_grid((64, 64))
    blob = np.exp(-((x + 3.0) ** 2 + (y - 1.0) ** 2) / (2 * 2.5**2))
    moved = ac.fourier_shift(blob, 2.3, -1.7)
    cy0, cx0 = ac.centroid(blob)
    cy1, cx1 = ac.centroid(moved)
    assert cx1 - cx0 == pytest.approx(2.3, abs=1e-6)
    assert cy1 - cy0 == pytest.approx(-1.7, abs=1e-6)


def test_shift_matches_the_tilt_convention() -> None:
    # An OPD ramp a x moves the image by +a (CONVENTIONS 3.1); on a periodic
    # (uncropped) FFT window a whole-pixel ramp is exactly a shifted image.
    pupil = ac.Pupil.circular(32, 1.0)
    lam, n_fft, k = 1e-6, 64, 3
    prop = ac.FFTPropagator(pupil.shape, n_fft, n_fft)
    pixel_scale = lam / (n_fft * pupil.pitch)
    y, x = pupil.coordinates()
    image = lambda opd: np.abs(prop(pupil.amplitude * np.exp(2j * np.pi * opd / lam))) ** 2  # noqa: E731
    flat = image(np.zeros(pupil.shape))
    tilt = k * pixel_scale
    np.testing.assert_allclose(ac.fourier_shift(flat, k, 0), image(tilt * x), atol=1e-12)
    np.testing.assert_allclose(ac.fourier_shift(flat, 0, -k), image(-tilt * y), atol=1e-12)


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_adjoint_is_the_opposite_shift(kind, rng) -> None:
    shape = (2, 12, 15)
    a, b = rng.standard_normal(shape), rng.standard_normal(shape)
    if kind == "complex":
        a = a + 1j * rng.standard_normal(shape)
        b = b + 1j * rng.standard_normal(shape)
    lhs = np.vdot(ac.fourier_shift(a, 0.3, -1.6), b)
    rhs = np.vdot(a, ac.fourier_shift(b, -0.3, 1.6))
    assert abs(lhs - rhs) < 1e-12 * abs(lhs)


def test_precision_and_validation(rng) -> None:
    a = rng.standard_normal((8, 8)).astype(np.float32)
    assert ac.fourier_shift(a, 0.5, 0.5).dtype == np.float32
    assert ac.fourier_shift(a.astype(np.complex64), 0.5, 0.0).dtype == np.complex64
    double = ac.get_backend("cpu", "double")
    assert ac.fourier_shift(a, 0.5, 0.5, backend=double).dtype == np.float64
    with pytest.raises(ValueError, match="2 dimensions"):
        ac.fourier_shift(np.ones(4), 1, 0)
    with pytest.raises(ValueError, match="finite"):
        ac.fourier_shift(a, np.inf, 0)


@pytest.mark.parametrize("device", devices())
def test_devices_agree(device, rng) -> None:
    a = rng.standard_normal((2, 20, 24))
    be = ac.get_backend(device, "double")
    out = ac.fourier_shift(be.asarray(a), 1.25, -0.5)
    assert type(out) is type(be.asarray(a))
    np.testing.assert_allclose(ac.to_numpy(out), ac.fourier_shift(a, 1.25, -0.5), atol=1e-12)
