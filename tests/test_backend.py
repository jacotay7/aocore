from __future__ import annotations

import numpy as np
import pytest

import aocore as sp
from aocore.backend import Backend, backend_of, get_backend, to_numpy
from conftest import devices


def test_cpu_defaults_to_double() -> None:
    be = get_backend("cpu")
    assert be.device == "cpu" and be.precision == "double"
    assert be.real_dtype == np.float64 and be.complex_dtype == np.complex128
    assert be.xp is np


def test_backends_are_cached_and_hashable() -> None:
    assert get_backend("cpu", "single") is get_backend("cpu", "float32")
    assert {get_backend("cpu"): 1}[get_backend("cpu", "double")] == 1


def test_existing_backend_passes_through_and_overrides() -> None:
    be = get_backend("cpu", "single")
    assert get_backend(be) is be
    assert get_backend(be, "double").precision == "double"


@pytest.mark.parametrize("bad", ["tpu", "cuda:7x"])
def test_unknown_device_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="device"):
        get_backend(bad)


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision"):
        get_backend("cpu", "half")


def test_auto_resolves_to_an_available_device() -> None:
    be = get_backend("auto")
    assert be.device == ("gpu" if sp.gpu_available() else "cpu")


def test_asarray_promotes_to_working_precision() -> None:
    be = get_backend("cpu", "single")
    assert be.asarray(np.ones(3)).dtype == np.float32
    assert be.asarray(np.ones(3, complex)).dtype == np.complex64
    assert be.asarray(np.ones(3, int)).dtype.kind == "i"
    assert be.asarray([1, 2], dtype="complex").dtype == np.complex64


@pytest.mark.parametrize("device", devices())
def test_unitary_fft_round_trip(device: str, rng: np.random.Generator) -> None:
    be = get_backend(device, "double")
    x = be.asarray(rng.standard_normal((3, 16, 12)) + 1j * rng.standard_normal((3, 16, 12)))
    y = be.fft2(x)
    assert np.isclose(be.dot(y, y), be.dot(x, x))
    np.testing.assert_allclose(to_numpy(be.ifft2(y)), to_numpy(x), atol=1e-12)


def test_dot_matches_numpy(rng: np.random.Generator) -> None:
    be = get_backend("cpu")
    a = rng.standard_normal(20000)
    b = rng.standard_normal(20000)
    assert np.isclose(be.dot(a, b), float(a @ b))
    za = a + 1j * b
    assert np.isclose(be.dot(za, za), float(np.sum(np.abs(za) ** 2)))


def test_backend_of_infers_precision() -> None:
    assert backend_of(np.zeros(2, np.float32)).precision == "single"
    assert backend_of(np.zeros(2)).precision == "double"
    assert isinstance(backend_of(np.zeros(2)), Backend)


@pytest.mark.gpu
def test_gpu_defaults_to_single_and_round_trips() -> None:
    be = get_backend("gpu")
    assert be.precision == "single"
    x = be.asarray(np.arange(4.0))
    assert type(x).__module__.startswith("cupy")
    np.testing.assert_array_equal(to_numpy(x), np.arange(4.0, dtype=np.float32))
    assert backend_of(x).device == "gpu"


def _full_padded_fft2(be, block, shape, out_shape, inverse=False):
    """The reference: zero-pad, full unitary transform, crop."""
    grid = be.zeros((*block.shape[:-2], *shape), "complex")
    grid[..., : block.shape[-2], : block.shape[-1]] = block
    spectrum = be.ifft2(grid) if inverse else be.fft2(grid)
    return spectrum[..., : out_shape[0], : out_shape[1]]


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize(
    ("block", "shape", "out_shape"),
    [
        ((32, 32), (64, 64), (32, 32)),
        ((31, 20), (64, 64), (17, 40)),
        ((50, 50), (120, 128), (60, 60)),
        ((33, 33), (97, 97), (40, 40)),
        ((64, 64), (64, 64), (64, 64)),
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_padded_fft2_matches_the_full_transform_to_the_last_bit(
    device, precision, block, shape, out_shape, inverse, rng
) -> None:
    be = get_backend(device, precision)
    x = be.asarray(rng.standard_normal((2, *block)) + 1j * rng.standard_normal((2, *block)))
    weights = be.asarray(np.exp(1j * rng.uniform(0, 6, block)), dtype="complex")
    out_weights = be.asarray(np.exp(1j * rng.uniform(0, 6, out_shape)), dtype="complex")
    ref = to_numpy(_full_padded_fft2(be, x * weights, shape, out_shape, inverse) * out_weights)
    got = be.padded_fft2(
        x, shape, out_shape, inverse=inverse, weights=weights, out_weights=out_weights
    )
    assert got.dtype == be.complex_dtype and got.shape == (2, *out_shape)
    # Rounding may differ in the last bit (SIMD remainder groups), never more.
    assert np.abs(to_numpy(got) - ref).max() <= 4 * be.eps * np.abs(ref).max()
    # Default crop, no weights, and writing into ``out``.
    plain = be.padded_fft2(x, shape, inverse=inverse)
    full = to_numpy(_full_padded_fft2(be, x, shape, shape, inverse))
    assert np.abs(to_numpy(plain) - full).max() <= 4 * be.eps * np.abs(full).max()
    out = be.zeros((2, *out_shape), "complex")
    assert be.padded_fft2(x, shape, out_shape, inverse=inverse, out=out) is out
    np.testing.assert_array_equal(
        to_numpy(out), to_numpy(be.padded_fft2(x, shape, out_shape, inverse=inverse))
    )


def test_padded_fft2_is_bitwise_on_whole_simd_bunches(monkeypatch, rng) -> None:
    # One thread, 16-line multiples: every line takes the full transform's path.
    monkeypatch.setenv("AOCORE_FFT_WORKERS", "1")
    be = get_backend("cpu", "double")
    x = rng.standard_normal((3, 48, 48)) + 1j * rng.standard_normal((3, 48, 48))
    for inverse in (False, True):
        np.testing.assert_array_equal(
            be.padded_fft2(x, (128, 128), (64, 64), inverse=inverse),
            _full_padded_fft2(be, x, (128, 128), (64, 64), inverse),
        )


def test_padded_fft2_rejects_blocks_outside_the_grid() -> None:
    be = get_backend("cpu")
    with pytest.raises(ValueError, match="must fit"):
        be.padded_fft2(np.zeros((8, 8)), (4, 8))
    with pytest.raises(ValueError, match="must fit"):
        be.padded_fft2(np.zeros((4, 4)), (8, 8), (4, 9))


def test_workspace_reuses_and_bounds_its_arrays() -> None:
    from aocore.backend import _workspace

    a = _workspace("test", (4, 4), np.float64)
    assert _workspace("test", (4, 4), np.float64) is a
    assert _workspace("test", (4, 4), np.complex128) is not a
    for i in range(8):
        _workspace(f"other {i}", (2,), np.float64)
    assert _workspace("test", (4, 4), np.float64) is not a  # evicted
    big = (65 * 2**20) // 8 + 1
    assert _workspace("big", (big,), np.float64) is not _workspace("big", (big,), np.float64)
