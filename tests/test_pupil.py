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
