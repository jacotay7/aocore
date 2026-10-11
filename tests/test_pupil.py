from __future__ import annotations

import numpy as np
import pytest

from aocore import Pupil


def test_circular_area_and_antialiasing() -> None:
    p = Pupil.circular(128, 2.0, obscuration=0.3, supersample=8)
    expected = np.pi * (1.0**2 - 0.3**2)
    assert abs(np.sum(p.amplitude) * p.pitch**2 - expected) / expected < 2e-3
    edge = p.amplitude[(p.amplitude > 0) & (p.amplitude < 1)]
    assert edge.size > 0  # grey rim pixels
    assert p.shape == (128, 128) and p.n_valid == int(np.count_nonzero(p.amplitude))


def test_spiders_block_light_symmetrically() -> None:
    p = Pupil.circular(128, 1.0, spiders=4, spider_width=0.03)
    full = Pupil.circular(128, 1.0)
    blocked = full.amplitude.sum() - p.amplitude.sum()
    assert blocked > 0
    np.testing.assert_allclose(p.amplitude, p.amplitude[::-1, ::-1], atol=1e-12)


@pytest.mark.parametrize(("factory", "segments"), [(Pupil.keck, 36), (Pupil.jwst, 18)])
def test_segmented_presets_have_labelled_segments(factory, segments) -> None:
    p = factory(128)
    assert p.n_segments == segments
    assert set(np.unique(p.segments)) == set(range(segments + 1))
    assert np.all(p.segments[p.amplitude == 0] == 0)


def test_pupil_validation_messages() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Pupil(-np.ones((4, 4)), pitch=1.0, diameter=1.0)
    with pytest.raises(ValueError, match="zero everywhere"):
        Pupil(np.zeros((4, 4)), pitch=1.0, diameter=1.0)
    with pytest.raises(ValueError, match="obscuration"):
        Pupil(np.ones((4, 4)), pitch=1.0, diameter=1.0, obscuration=1.2)


def test_from_array_infers_pitch_from_diameter() -> None:
    amp = np.zeros((10, 10))
    amp[2:8, 2:8] = 1
    p = Pupil.from_array(amp, diameter=3.0)
    assert np.isclose(p.pitch, 0.5)


def test_padding_and_downsampling_preserve_light() -> None:
    p = Pupil.circular(64, 1.0)
    assert np.isclose(p.padded(96).amplitude.sum(), p.amplitude.sum())
    d = p.downsampled(2)
    assert d.shape == (32, 32) and np.isclose(d.pitch, 2 * p.pitch)
    assert np.isclose(d.amplitude.sum() * 4, p.amplitude.sum())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Pupil.circular(96, 2.0, obscuration=0.3, spiders=4, spider_width=0.04, fill=0.8),
        lambda: Pupil.keck(96).padded(112),
        lambda: Pupil.from_array(Pupil.circular(64, 1.0, fill=0.8).amplitude, pitch=1 / 51.2),
    ],
    ids=["circular", "keck", "array"],
)
def test_shifted_pupil_conserves_area_and_moves_towards_positive_x(factory) -> None:
    pupil = factory()
    assert pupil.shifted(0.0, 0.0) is pupil
    dx, dy = 0.37 * pupil.pitch, -1.21 * pupil.pitch
    moved = pupil.shifted(dx, dy)
    assert moved.shape == pupil.shape and moved.pitch == pupil.pitch
    assert moved.diameter == pupil.diameter
    # Area (the sum of area fractions) to within the anti-aliasing error, which
    # the 1.5-pixel, grid-aligned spider vanes dominate; exact for the array
    # resampling.
    area, moved_area = pupil.amplitude.sum(), moved.amplitude.sum()
    assert abs(moved_area - area) / area < 4e-3
    y, x = pupil.coordinates()

    def centre(p: Pupil) -> tuple[float, float]:
        w = p.amplitude
        return float((w * y).sum() / w.sum()), float((w * x).sum() / w.sum())

    (y0, x0), (y1, x1) = centre(pupil), centre(moved)
    assert abs((x1 - x0) - dx) < 0.03 * pupil.pitch  # +x = larger column index
    assert abs((y1 - y0) - dy) < 0.03 * pupil.pitch
    if pupil.segments is not None:
        assert moved.n_segments == pupil.n_segments
        assert np.all((moved.segments > 0) == (moved.amplitude > 0))


def test_shifted_analytic_pupil_whole_pixels_equals_a_roll_and_keeps_segments() -> None:
    pupil = Pupil.keck(96).padded(104)
    moved = pupil.shifted(3 * pupil.pitch, -2 * pupil.pitch)
    np.testing.assert_array_equal(moved.amplitude, np.roll(pupil.amplitude, (-2, 3), (0, 1)))
    np.testing.assert_array_equal(moved.segments, np.roll(pupil.segments, (-2, 3), (0, 1)))
    # Re-rendered, not resampled: rim pixels stay as sharp as the original's.
    sub = pupil.shifted(0.5 * pupil.pitch, 0.0)
    grey = lambda p: int(np.count_nonzero((p.amplitude > 0) & (p.amplitude < 1)))  # noqa: E731
    assert grey(sub) < 1.2 * grey(pupil)
    finer = pupil.shifted(0.5 * pupil.pitch, 0.0, supersample=8)
    area = pupil.amplitude.sum()
    assert abs(finer.amplitude.sum() - area) < 1e-3 * area


def test_shifted_array_pupil_is_an_area_weighted_resample() -> None:
    amp = np.zeros((8, 8))
    amp[3:5, 3:5] = 1.0
    pupil = Pupil.from_array(amp, pitch=0.5, segments=(amp > 0).astype(int))
    moved = pupil.shifted(0.25 * 0.5, 0.0)  # a quarter pixel along +x
    np.testing.assert_allclose(moved.amplitude[3, 3:6], [0.75, 1.0, 0.25])
    assert moved.amplitude.sum() == pytest.approx(4.0)
    assert set(np.unique(moved.segments)) == {0, 1}
    edge = pupil.shifted(-2.25, 0.0)  # 4.5 pixels: three quarters leave the grid
    assert edge.amplitude.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite"):
        pupil.shifted(float("nan"), 0.0)
    with pytest.raises(ValueError, match="zero everywhere"):
        pupil.shifted(10.0, 0.0)
    # Derived pupils lose the analytic geometry and fall back to resampling.
    circ = Pupil.circular(32, 1.0, fill=0.75)
    assert circ.with_amplitude(circ.amplitude)._geometry is None
    assert circ.downsampled(2)._geometry is None and circ.padded(40)._geometry is not None
