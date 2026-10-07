from __future__ import annotations

import numpy as np
import pytest

import aocore as ac
from aocore import Pupil
from aocore.unwrap import unwrap_phase, wrap
from conftest import devices


def _smooth_phase(pupil: Pupil, scale: float) -> np.ndarray:
    """A smooth polynomial phase (radians) over the pupil, with ~scale rad RMS."""
    y, x = pupil.coordinates()
    r = pupil.diameter / 2.0
    u, v = x / r, y / r
    phase = 1.3 * (2 * (u**2 + v**2) - 1) + 0.8 * (u**2 - v**2) + 0.6 * u * v * 2 + 0.5 * u**3
    phase = phase * scale
    return np.where(pupil.mask, phase, 0.0)


@pytest.mark.parametrize(
    "pupil",
    [Pupil.circular(96, 1.0, obscuration=0.3, spiders=4, spider_width=0.02), Pupil.keck(96)],
    ids=["spiders", "keck"],
)
@pytest.mark.parametrize("device", devices())
def test_unwrap_recovers_smooth_phase(pupil, device) -> None:
    phase = _smooth_phase(pupil, 6.0)
    be = ac.get_backend(device, "double")
    out = ac.to_numpy(unwrap_phase(be.asarray(wrap(phase)), pupil.mask, weights=pupil.amplitude))
    diff = (out - phase)[pupil.mask]
    assert np.ptp(phase[pupil.mask]) > 4 * np.pi
    assert np.std(diff) < 1e-6
    assert abs(wrap(np.mean(diff))) < 1e-6


def test_unwrap_bridges_disconnected_regions() -> None:
    pupil = Pupil.circular(96, 1.0, obscuration=0.2, spiders=3, spider_width=0.04)
    y, x = pupil.coordinates()
    phase = 40.0 * x + 25.0 * y
    out = unwrap_phase(wrap(phase), pupil.mask, weights=pupil.amplitude)
    assert np.std((out - phase)[pupil.mask]) < 1e-6


def test_wrap_range() -> None:
    v = wrap(np.linspace(-20, 20, 1001))
    assert v.min() >= -np.pi - 1e-12 and v.max() <= np.pi + 1e-12


def test_metrics_and_mode_removal() -> None:
    pupil = Pupil.circular(64, 1.0)
    _, x = pupil.coordinates()
    tilt = np.where(pupil.mask, 1e-7 * x / 0.5 + 3e-8, 0.0)
    assert ac.rms(tilt, pupil, "tiptilt") < 1e-20
    assert ac.rms(tilt, pupil, "piston") > 1e-8
    assert ac.rms(tilt, pupil.amplitude) == pytest.approx(ac.rms(tilt, pupil))
    assert np.isclose(ac.strehl_from_rms(1e-6 / (2 * np.pi), 1e-6), np.exp(-1))
    assert ac.strehl_marechal(0.0, 1e-6) == 1.0
    keck = Pupil.keck(64)
    pistons = np.zeros(keck.shape)
    for s in range(1, keck.n_segments + 1):
        pistons[keck.segments == s] = 1e-8 * s
    assert ac.rms(pistons, keck, "segment_piston") < 1e-20
    with pytest.raises(ValueError, match="unknown mode"):
        ac.rms(np.zeros(pupil.shape), pupil, "coma")


def test_wavefront_error_handles_the_twin() -> None:
    pupil = Pupil.circular(64, 1.0)
    z = _smooth_phase(pupil, 1.0) * 1e-8
    twin = -z[::-1, ::-1]
    assert ac.wavefront_error(twin, z, pupil, allow_twin=True) < 1e-15
    assert ac.wavefront_error(twin, z, pupil) > 1e-9


def test_block_sum_conserves_flux() -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(size=(2, 12, 18))
    binned = ac.block_sum(image, 3)
    assert binned.shape == (2, 4, 6)
    np.testing.assert_allclose(binned.sum(axis=(1, 2)), image.sum(axis=(1, 2)))
    assert ac.block_sum(image, (2, 3)).shape == (2, 6, 6)
    with pytest.raises(ValueError, match="multiple"):
        ac.block_sum(image, 5)


def test_aocore_primitives_pass_their_own_conformance_checks() -> None:
    from aocore import conformance as cf

    cf.check_coordinates(ac.centered_coordinates)
    cf.check_rms(lambda opd, amp: ac.rms(opd, amp))
    pupil = Pupil.circular(32, 1.0)
    lam, q = 1e-6, 4
    pixel = lam / pupil.diameter / q
    prop = ac.FocalPlanePropagator(pupil.shape, pupil.pitch, lam, pixel, (48, 48))
    norm = float(np.sum(pupil.amplitude**2))

    def image(opd: np.ndarray) -> np.ndarray:
        u = pupil.amplitude * np.exp(2j * np.pi * opd / lam)
        return np.abs(prop.forward(u[None])[0]) ** 2 / norm

    cf.check_image_centring(image, pupil_shape=pupil.shape)
    cf.check_tilt_direction(image, pupil_shape=pupil.shape, pitch=pupil.pitch, pixel_scale=pixel)
    full = ac.FocalPlanePropagator(pupil.shape, pupil.pitch, lam, pixel, (128, 128))
    cf.check_unit_flux(
        lambda opd: (
            np.abs(full.forward((pupil.amplitude * np.exp(2j * np.pi * opd / lam))[None])[0]) ** 2
            / norm
        ),
        pupil_shape=pupil.shape,
    )


def _grey_pupil_and_opd() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    amplitude = Pupil.circular(48, 1.0, obscuration=0.2).amplitude.copy()
    amplitude[amplitude > 0] *= rng.uniform(0.3, 1.0, int(np.count_nonzero(amplitude)))
    y, x = ac.coordinate_grid(amplitude.shape, 1.0 / 48)
    opd = 5e-7 + 2e-7 * x - 3e-7 * y + 1e-8 * rng.standard_normal(amplitude.shape)
    return opd, amplitude


@pytest.mark.parametrize("device", devices())
def test_rms_unweighted_keeps_piston_and_ignores_weights(device) -> None:
    be = ac.get_backend(device, "double")
    opd, amplitude = _grey_pupil_and_opd()
    mask = amplitude > 0
    expected = float(np.sqrt(np.mean(opd[mask] ** 2)))
    assert ac.rms_unweighted(be.asarray(opd), be.asarray(mask)) == pytest.approx(expected)
    # Non-zero amplitude counts as inside, whatever its value; a Pupil uses its mask.
    assert ac.rms_unweighted(be.asarray(opd), amplitude) == pytest.approx(expected)
    pupil = Pupil.from_array(amplitude)
    assert ac.rms_unweighted(be.asarray(opd), pupil) == pytest.approx(expected)
    # Piston is not removed: a constant c gives |c|.
    assert ac.rms_unweighted(be.asarray(np.full((4, 4), -3e-7))) == pytest.approx(3e-7)
    # NaN outside the mask is ignored; float32 input is accumulated in float64.
    holes = np.where(mask, opd, np.nan).astype(np.float32)
    got = ac.rms_unweighted(be.asarray(holes, np.float32), mask)
    assert got == pytest.approx(expected, rel=1e-6)


def test_rms_unweighted_errors() -> None:
    with pytest.raises(ValueError, match="shape"):
        ac.rms_unweighted(np.zeros((3, 3)), np.ones((4, 4)))
    with pytest.raises(ValueError, match="mask is empty"):
        ac.rms_unweighted(np.zeros((3, 3)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="empty"):
        ac.rms_unweighted(np.zeros(0))


@pytest.mark.parametrize("device", devices())
def test_rms_tiptilt_removed_matches_remove_modes(device) -> None:
    be = ac.get_backend(device, "double")
    opd, amplitude = _grey_pupil_and_opd()
    pupil = Pupil.from_array(amplitude, pitch=1.0 / 48)
    expected = ac.rms(opd, pupil, "tiptilt")
    assert 5e-9 < expected < 2e-8  # the noise, not the piston and tilts
    got = ac.rms_tiptilt_removed(be.asarray(opd), be.asarray(amplitude))
    assert got == pytest.approx(expected, rel=1e-10)
    assert ac.rms_tiptilt_removed(be.asarray(opd), pupil) == pytest.approx(expected, rel=1e-10)
    holes = np.where(amplitude > 0, opd, np.nan)
    assert ac.rms_tiptilt_removed(be.asarray(holes), pupil) == pytest.approx(expected, rel=1e-10)
    y, x = ac.coordinate_grid(pupil.shape)
    plane = be.asarray(1e-6 + 3e-7 * x + 2e-7 * y)
    assert ac.rms_tiptilt_removed(plane, pupil) < 1e-20


def test_rms_tiptilt_removed_edge_cases() -> None:
    # One illuminated row: tilt is undetermined, and the fit stays finite.
    amplitude = np.zeros((5, 6))
    amplitude[2] = 1.0
    opd = np.tile(np.arange(6.0), (5, 1)) + 7.0
    assert ac.rms_tiptilt_removed(opd, amplitude) < 1e-12
    with pytest.raises(ValueError, match="2-D"):
        ac.rms_tiptilt_removed(np.zeros((3, 3)), np.ones((4, 4)))
    with pytest.raises(ValueError, match="transmission"):
        ac.rms_tiptilt_removed(np.zeros((3, 3)), np.zeros((3, 3)))


def _block_sum_reference(array: np.ndarray, fy: int, fx: int) -> np.ndarray:
    *lead, ny, nx = array.shape
    return array.reshape(*lead, ny // fy, fy, nx // fx, fx).sum(axis=(-3, -1))


@pytest.mark.parametrize("factor", [2, 3, 4, 5, 8, (2, 3), (1, 6), (5, 1), (4, 8)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128, np.uint8, np.int16, bool])
def test_block_sum_matches_the_two_axis_reduction(factor, dtype) -> None:
    rng = np.random.default_rng(3)
    fy, fx = (factor, factor) if isinstance(factor, int) else factor
    shape = (3, 4 * fy * 3, 4 * fx * 5)
    if np.dtype(dtype).kind == "c":
        image = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    else:
        image = (rng.uniform(0, 255, shape)).astype(dtype)
    expected = _block_sum_reference(image, fy, fx)
    for data in (image, np.asfortranarray(image)):
        got = ac.block_sum(data, factor)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        if np.dtype(dtype).kind in "biu":
            np.testing.assert_array_equal(got, expected)  # exact, no overflow
        else:
            rtol = 1e-5 if dtype == np.float32 else 1e-13
            np.testing.assert_allclose(got, expected, rtol=rtol)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize(
    ("shape", "factor"),
    [((24, 24), 2), ((24, 24), 4), ((24, 24), 8), ((384, 384), 3), ((512, 512), (4, 2))],
)
def test_block_sum_on_each_backend(device, shape, factor) -> None:
    be = ac.get_backend(device, "single")
    rng = np.random.default_rng(4)
    image = rng.uniform(size=shape).astype(np.float32)
    fy, fx = (factor, factor) if isinstance(factor, int) else factor
    got = ac.block_sum(be.asarray(image), factor)
    assert isinstance(got, type(be.asarray(image)))
    np.testing.assert_allclose(ac.to_numpy(got), _block_sum_reference(image, fy, fx), rtol=1e-5)
    counts = be.asarray(rng.integers(0, 255, shape).astype(np.uint8))
    host = ac.to_numpy(counts)
    np.testing.assert_array_equal(
        ac.to_numpy(ac.block_sum(counts, factor)), _block_sum_reference(host, fy, fx)
    )
    mean = ac.block_mean(be.asarray(image), factor)
    np.testing.assert_allclose(
        ac.to_numpy(mean), _block_sum_reference(image, fy, fx) / (fy * fx), rtol=1e-5
    )


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("dtype", [np.float64, np.complex64, np.int16, np.uint32, bool])
def test_block_sum_dtypes_and_layouts_on_each_backend(device, dtype) -> None:
    be = ac.get_backend(device)
    rng = np.random.default_rng(5)
    image = rng.uniform(-100, 100, (2, 3, 12, 30))
    if np.dtype(dtype).kind == "c":
        image = image + 1j * rng.uniform(-100, 100, image.shape)
    elif np.dtype(dtype).kind in "bu":
        image = np.abs(image)
    image = image.astype(dtype)
    for data, factor in ((image, (3, 5)), (np.swapaxes(image, -1, -2)[..., :, :12], 6)):
        host = np.asarray(data)
        expected = _block_sum_reference(host, *_factor_pair(factor))
        got = ac.block_sum(be.xp.asarray(host) if device == "gpu" else data, factor)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        np.testing.assert_allclose(ac.to_numpy(got), expected, rtol=1e-6, atol=1e-3)


def _factor_pair(factor: int | tuple[int, int]) -> tuple[int, int]:
    return (factor, factor) if isinstance(factor, int) else factor


def test_block_mean() -> None:
    counts = np.arange(16, dtype=np.int32).reshape(4, 4)
    np.testing.assert_array_equal(ac.block_mean(counts, 2), [[2.5, 4.5], [10.5, 12.5]])
    assert ac.block_mean(counts, 1).dtype == np.float64
    with pytest.raises(ValueError, match=">= 1"):
        ac.block_mean(counts, (1, 0))
