"""Perfect coronagraph, azimuthal profiles and strehl_peak (CONVENTIONS 3.3, 4.2)."""

from __future__ import annotations

import numpy as np
import pytest

import aocore as ac
from conftest import devices

LAM = 1.6e-6


def _setup(device: str = "cpu") -> tuple[ac.Pupil, ac.FocalPlanePropagator]:
    pupil = ac.Pupil.circular(48, 8.0, obscuration=0.14, fill=0.9)
    prop = ac.FocalPlanePropagator(
        pupil.shape, pupil.pitch, LAM, LAM / pupil.diameter / 3, (96, 96), backend=device
    )
    return pupil, prop


@pytest.mark.parametrize("device", devices())
def test_flat_wavefront_is_fully_rejected(device) -> None:
    pupil, prop = _setup(device)
    image = ac.perfect_coronagraph_image(np.zeros(pupil.shape), pupil, prop)
    assert image.shape == (1, 96, 96)
    assert float(ac.to_numpy(image).max()) < 1e-12
    # The same for a complex field with the pupil amplitude.
    field = pupil.amplitude[None].astype(complex)
    assert float(ac.to_numpy(ac.perfect_coronagraph_image(field, pupil, prop)).max()) < 1e-12


def test_halo_carries_the_light_the_coronagraph_cannot_remove(rng) -> None:
    pupil = ac.Pupil.circular(32, 1.0, fill=0.75)
    n_fft = 128
    prop = ac.FFTPropagator(pupil.shape, n_fft, n_fft)  # captures all the light
    opd = rng.standard_normal((3, *pupil.shape)) * 30e-9
    field = pupil.amplitude * np.exp(2j * np.pi * opd / LAM)
    image = ac.perfect_coronagraph_image(opd, pupil, prop, wavelength=LAM, normalize="flux")
    a = pupil.amplitude
    on_axis_strehl = np.abs((a * field).sum(axis=(-2, -1))) ** 2 / (a**2).sum() ** 2
    # Unitary propagation: the halo holds 1 - (on-axis Strehl) of the light, per frame.
    np.testing.assert_allclose(image.sum(axis=(-2, -1)), 1 - on_axis_strehl, rtol=1e-10)
    raw = ac.perfect_coronagraph_image(field, pupil, prop, normalize=None)
    np.testing.assert_allclose(raw, image * (a**2).sum(), rtol=1e-10)
    peak = ac.perfect_coronagraph_image(field, pupil, prop, normalize="peak")
    reference = np.abs(prop(a.astype(complex))) ** 2
    np.testing.assert_allclose(peak, raw / reference.max(), rtol=1e-10)
    # Without a coronagraph the same window holds all the light.
    psf = np.abs(prop(field)) ** 2
    assert np.all(peak.max(axis=(-2, -1)) < 0.1 * psf.max(axis=(-2, -1)) / reference.max())


def test_broadband_opd_gets_one_slice_per_wavelength(rng) -> None:
    pupil = ac.Pupil.circular(32, 4.0, fill=0.8)
    wavelengths = [1.5e-6, 1.65e-6]
    prop = ac.FocalPlanePropagator(pupil.shape, pupil.pitch, wavelengths, 1.5e-6 / 4 / 3, 48)
    opd = rng.standard_normal((5, *pupil.shape)) * 20e-9
    image = ac.perfect_coronagraph_image(opd, pupil, prop)
    assert image.shape == (5, 2, 48, 48)
    one = ac.FocalPlanePropagator(pupil.shape, pupil.pitch, wavelengths[1], 1.5e-6 / 4 / 3, 48)
    np.testing.assert_allclose(image[:, 1:], ac.perfect_coronagraph_image(opd, pupil, one))
    with pytest.raises(ValueError, match="wavelengths"):
        ac.perfect_coronagraph_image(opd, pupil, prop, wavelength=1e-6)


def test_coherent_field_removal_is_an_orthogonal_projection(rng) -> None:
    pupil = ac.Pupil.circular(24, 1.0, obscuration=0.2)
    shape = (2, *pupil.shape)
    u = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    v = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    pu = ac.perfect_coronagraph_field(u, pupil)
    # Self-adjoint (its own exact adjoint) and idempotent.
    lhs, rhs = np.vdot(pu, v), np.vdot(u, ac.perfect_coronagraph_field(v, pupil))
    assert abs(lhs - rhs) < 1e-12 * abs(lhs)
    np.testing.assert_allclose(ac.perfect_coronagraph_field(pu, pupil), pu, atol=1e-12)
    assert abs((pupil.amplitude * pu).sum(axis=(-2, -1))).max() < 1e-10


def test_validation() -> None:
    pupil, prop = _setup()
    with pytest.raises(ValueError, match="normalize"):
        ac.perfect_coronagraph_image(np.zeros(pupil.shape), pupil, prop, normalize="max")
    with pytest.raises(ValueError, match="propagator input"):
        ac.perfect_coronagraph_image(np.zeros((8, 8)), np.ones((8, 8)), prop)
    fft = ac.FFTPropagator(pupil.shape, 96, 96)
    with pytest.raises(ValueError, match="wavelength"):
        ac.perfect_coronagraph_image(np.zeros(pupil.shape), pupil, fft)
    with pytest.raises(ValueError, match="pupil shape"):
        ac.perfect_coronagraph_field(np.zeros((4, 4), complex), pupil)
    with pytest.raises(ValueError, match="transmission"):
        ac.perfect_coronagraph_field(np.zeros((4, 4), complex), np.zeros((4, 4)))


def test_azimuthal_profile_bins_by_radius_about_the_optical_axis() -> None:
    y, x = ac.coordinate_grid((20, 20))
    r = np.hypot(x, y)
    radii, values = ac.azimuthal_profile(r)
    np.testing.assert_allclose(radii, np.arange(radii.size) + 0.5)
    assert radii.size == int(np.ceil(r.max()))
    # The mean radius of the pixels in [k, k + 1) lies inside the bin.
    assert np.all((values >= radii - 0.5) & (values < radii + 0.5))
    # A shifted centre and lambda/D units (3 pixels per lambda/D).
    cy, cx = 2.0, -3.0
    ring = (np.abs(np.hypot(x - cx, y - cy) - 7.5) < 1.5).astype(float)  # 2-3 lambda/D
    radii, values = ac.azimuthal_profile(ring, (cy, cx), 1.0, pixel_scale=1 / 3, max_radius=4.0)
    np.testing.assert_allclose(radii, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(values[:3], [0.0, 0.0, 1.0])


def test_azimuthal_profile_frames_empty_bins_and_errors() -> None:
    stack = np.stack([np.ones((9, 9)), 2 * np.ones((9, 9))])
    radii, values = ac.azimuthal_profile(stack, bin_width=0.25)
    assert values.shape == (2, radii.size)
    assert np.isnan(values[0, 2])  # no pixel centre between 0.5 and 0.75
    np.testing.assert_allclose(values[1][~np.isnan(values[1])], 2.0)
    with pytest.raises(ValueError, match="2-D"):
        ac.azimuthal_profile(np.ones(5))
    with pytest.raises(ValueError, match="positive"):
        ac.azimuthal_profile(np.ones((5, 5)), bin_width=0)


def test_strehl_peak_is_the_peak_ratio_with_the_unaberrated_image(rng) -> None:
    pupil, prop = _setup()
    a = pupil.amplitude.astype(complex)
    _, x = pupil.coordinates()
    reference = np.abs(prop(a[None]))[0] ** 2
    tilted = np.abs(prop((a * np.exp(2j * np.pi * 0.4 * x / pupil.diameter))[None]))[0] ** 2
    assert ac.strehl_peak(reference, reference) == 1.0
    # Tilt moves the peak; the brightest pixel still counts.
    assert ac.strehl_peak(tilted, reference) > 0.95
    assert ac.strehl_peak(tilted, float(reference.max())) == ac.strehl_peak(tilted, reference)
    opd = rng.standard_normal((4, *pupil.shape)) * 25e-9
    frames = np.abs(prop((a * np.exp(2j * np.pi * opd / LAM))[:, None]))[:, 0] ** 2
    ratios = ac.strehl_peak(frames, reference)
    assert ratios.shape == (4,)
    marechal = np.array([ac.strehl_marechal(ac.rms(o, pupil), LAM) for o in opd])
    np.testing.assert_allclose(ratios, marechal, atol=0.02)
    renormalized = ac.strehl_peak(7 * frames, reference, normalize=True)
    np.testing.assert_allclose(renormalized, ratios, rtol=0.05)
    with pytest.raises(ValueError, match="2-D"):
        ac.strehl_peak(np.ones(3), reference)
    with pytest.raises(ValueError, match="reference image"):
        ac.strehl_peak(frames, 1.0, normalize=True)


@pytest.mark.parametrize("device", devices())
def test_devices_agree(device, rng) -> None:
    pupil, prop_cpu = _setup()
    _, prop = _setup(device)
    opd = rng.standard_normal((2, *pupil.shape)) * 30e-9
    be = ac.get_backend(device)
    image = ac.perfect_coronagraph_image(be.asarray(opd), pupil, prop)
    expected = ac.perfect_coronagraph_image(opd, pupil, prop_cpu)
    np.testing.assert_allclose(ac.to_numpy(image), expected, rtol=1e-3, atol=1e-9)
    _, values = ac.azimuthal_profile(image)
    _, cpu_values = ac.azimuthal_profile(expected)
    np.testing.assert_allclose(ac.to_numpy(values), cpu_values, rtol=1e-3, atol=1e-9)
    psf = be.asarray(np.abs(prop_cpu(pupil.amplitude[None].astype(complex)))[0] ** 2)
    assert ac.strehl_peak(psf, psf) == pytest.approx(1.0)
