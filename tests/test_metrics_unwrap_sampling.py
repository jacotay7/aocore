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
