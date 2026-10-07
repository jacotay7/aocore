"""Backend, conventions, sampling and propagation paths not covered elsewhere."""

from __future__ import annotations

import math

import numpy as np
import pytest

import aocore as ac
from aocore.backend import _blas_threads, _cpu_workers
from conftest import devices


def test_conventions_helpers() -> None:
    lam = 1e-6
    assert pytest.approx(1.0) == ac.ARCSEC_TO_RAD * ac.RAD_TO_ARCSEC
    assert ac.opd_to_phase(lam / 2, lam) == pytest.approx(math.pi)
    assert ac.phase_to_opd(math.pi, lam) == pytest.approx(lam / 2)
    y, x = ac.coordinate_grid((3, 4), 2.0)
    assert x[0].tolist() == [-3.0, -1.0, 1.0, 3.0] and y[:, 0].tolist() == [-2.0, 0.0, 2.0]
    image = np.zeros((5, 5))
    image[1, 3] = 1.0
    assert ac.centroid(image) == (-1.0, 1.0)
    with pytest.raises(ValueError):
        ac.centroid(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        ac.centered_coordinates(0)
    with pytest.raises(ValueError, match="shape"):
        ac.conventions.rms(np.zeros(3), np.ones(4))
    with pytest.raises(ValueError, match="transmission"):
        ac.conventions.rms(np.zeros(3), np.zeros(3))


def test_backend_fft_helpers_and_threading(monkeypatch) -> None:
    be = ac.get_backend("cpu", "double")
    rng = np.random.default_rng(0)
    real = rng.standard_normal((2, 8, 6))
    back = be.irfft2(be.rfft2(real), (8, 6))
    np.testing.assert_allclose(back, real, atol=1e-12)
    with be.blas_limit(1e9):
        prod = real[0] @ real[0].T
    assert prod.shape == (8, 8)
    assert be.dot(real + 1j, real + 1j) == pytest.approx(float(np.sum(np.abs(real + 1j) ** 2)))
    assert be.scalar(np.float64(2.5)) == 2.5
    be.synchronize()
    assert be.random(3).integers(10) == np.random.default_rng(3).integers(10)
    assert be.empty((2, 2), "complex").dtype == np.complex128
    assert be.zeros(3, np.int32).dtype == np.int32
    assert _cpu_workers(10) == 1
    monkeypatch.setenv("AOCORE_FFT_WORKERS", "3")
    assert _cpu_workers(10) == 3
    monkeypatch.setenv("AOCORE_BLAS_THREADS", "2")
    assert _blas_threads(1e12) == 2
    monkeypatch.delenv("AOCORE_BLAS_THREADS")
    assert 1 <= _blas_threads(1e12) <= 8
    assert repr(be) == "Backend(device='cpu', precision='double')"
    assert ac.get_backend(None).device == "cpu"
    assert ac.get_backend("cuda" if ac.gpu_available() else "cpu").device in ("cpu", "gpu")


def test_block_sum_identity_and_errors() -> None:
    a = np.ones((4, 4))
    assert ac.block_sum(a, 1) is a
    with pytest.raises(ValueError, match=">= 1"):
        ac.block_sum(a, 0)


def test_propagator_validation() -> None:
    with pytest.raises(ValueError, match="positive int"):
        ac.FFTPropagator((0, 4), 4, 8)
    with pytest.raises(ValueError, match="scalar or a"):
        ac.FFTPropagator(4, 4, 8, offset=(1, 2, 3))
    with pytest.raises(ValueError, match="positive and finite"):
        ac.MFTPropagator(4, 4, -2.0)
    with pytest.raises(ValueError, match="stacked samples"):
        ac.MFTPropagator(4, 4, np.ones((2, 3)), stack=True)
    with pytest.raises(ValueError, match="scalar or"):
        ac.MFTPropagator(4, 4, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        ac.FocalPlanePropagator(4, 0.1, -1.0, 0.1, 4)
    with pytest.raises(ValueError, match="pupil_pitch"):
        ac.FocalPlanePropagator(4, 0.0, 1.0, 0.1, 4)
    with pytest.raises(ValueError, match="method"):
        ac.FocalPlanePropagator(4, 0.1, 1.0, 0.1, 4, method="magic")
    prop = ac.MFTPropagator(4, 4, (8.0, 9.0))
    assert prop(np.ones((4, 4), complex)).shape == (4, 4)
    asp = ac.AngularSpectrumPropagator(8, 1e-5, 1e-6, 1e-3, paraxial=True)
    u = np.ones((8, 8), complex)
    np.testing.assert_allclose(np.abs(asp.forward(u)), 1.0, atol=1e-12)


@pytest.mark.parametrize("device", devices())
def test_coordinate_helpers_dtype_and_backend(device) -> None:
    host = ac.centered_coordinates(6, 0.25, 0.5)
    assert isinstance(host, np.ndarray) and host.dtype == np.float64  # default unchanged
    be = ac.get_backend(device)
    coords = ac.centered_coordinates(6, 0.25, 0.5, backend=device)
    assert coords.dtype == be.real_dtype and isinstance(coords, type(be.zeros(1)))
    np.testing.assert_array_equal(ac.to_numpy(coords), host.astype(be.real_dtype))
    single = ac.centered_coordinates(5, 0.1, dtype=np.float32, backend=be)
    np.testing.assert_array_equal(
        ac.to_numpy(single), ac.centered_coordinates(5, 0.1).astype(np.float32)
    )
    y, x = ac.coordinate_grid((3, 4), 2.0, dtype=np.float32, backend=device)
    assert y.shape == x.shape == (3, 4) and x.dtype == np.float32
    assert ac.to_numpy(x)[0].tolist() == [-3.0, -1.0, 1.0, 3.0]
    assert ac.to_numpy(y)[:, 0].tolist() == [-2.0, 0.0, 2.0]
    gy, gx = ac.coordinate_grid(4)
    assert gy.dtype == np.float64 and not gx.flags.writeable
