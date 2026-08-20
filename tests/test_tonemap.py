"""Tone mapping operators and gamut compression."""

import numpy as np
import pytest

from vct.core.tonemap import (DIFFUSE_WHITE_NITS, TONE_MAPPERS, gamut_compress,
                              hlg_system_gamma, normalise_hdr, tone_map)

PEAK_1000 = 1000.0 / DIFFUSE_WHITE_NITS


@pytest.mark.parametrize("mapper_id", sorted(TONE_MAPPERS))
def test_black_stays_black(mapper_id):
    """Any operator that lifts black turns every shadow milky."""
    out = TONE_MAPPERS[mapper_id].fn(np.array([0.0]), PEAK_1000)
    assert abs(float(out[0])) < 1e-6


@pytest.mark.parametrize("mapper_id", sorted(TONE_MAPPERS))
def test_monotonic(mapper_id):
    """A non-monotonic curve would invert contrast somewhere in the image."""
    x = np.linspace(0.0, PEAK_1000, 2000)
    y = np.asarray(TONE_MAPPERS[mapper_id].fn(x, PEAK_1000))
    assert np.all(np.diff(y) >= -1e-6)
    assert np.isfinite(y).all()


@pytest.mark.parametrize("mapper_id", ["bt2390", "hable", "reinhard", "mobius"])
def test_peak_maps_to_white(mapper_id):
    """The whole point: source peak must land on display white, not above it."""
    out = float(TONE_MAPPERS[mapper_id].fn(np.array([PEAK_1000]), PEAK_1000)[0])
    assert abs(out - 1.0) < 1e-3


@pytest.mark.parametrize("mapper_id", sorted(TONE_MAPPERS))
def test_midtones_are_not_crushed(mapper_id):
    """Mid grey must survive an HDR conversion recognisably.

    This is the failure this tool exists to avoid: normalising by peak instead of
    by diffuse white drags 18% grey down to a few percent and the result looks
    muddy. Half a stop of darkening is a look; three stops is a bug.
    """
    out = float(TONE_MAPPERS[mapper_id].fn(np.array([0.18]), PEAK_1000)[0])
    assert 0.18 / 2.4 < out <= 0.181, f"{mapper_id} put 18% grey at {out:.4f}"


def test_bt2390_passes_low_values_through_untouched():
    """The EETF's knee means everything below it is bit-exact."""
    x = np.linspace(0.0, 0.15, 50)
    assert np.allclose(TONE_MAPPERS["bt2390"].fn(x, PEAK_1000), x, atol=1e-4)


def test_clip_is_the_null_operator():
    x = np.linspace(0.0, 1.0, 50)
    assert np.allclose(TONE_MAPPERS["clip"].fn(x, PEAK_1000), x)


def test_pq_normalisation_uses_diffuse_white():
    """BT.2408: 203 cd/m2 is 1.0 after normalisation, not 10 000."""
    linear = np.array([[203.0 / 10000.0] * 3])
    out = normalise_hdr(linear, "pq")
    assert np.allclose(out, 1.0, atol=1e-6)


def test_hlg_normalisation_puts_reference_white_at_one():
    """HLG 75% signal through the OOTF must also land on diffuse white."""
    from vct.core.transfer import hlg_decode
    scene = np.full((1, 3), float(hlg_decode(np.array(0.75))), dtype=np.float32)
    out = normalise_hdr(scene, "hlg", source_peak_nits=1000.0, primaries="bt2020")
    assert np.allclose(out, 1.0, atol=2e-3), out


def test_hlg_system_gamma_matches_bt2100():
    assert abs(hlg_system_gamma(1000.0) - 1.2) < 1e-9
    assert hlg_system_gamma(4000.0) > 1.2   # brighter display, stronger OOTF


def test_sdr_normalisation_is_a_no_op():
    x = np.random.default_rng(0).random((8, 3)).astype(np.float32)
    assert np.array_equal(normalise_hdr(x, "bt709"), x)


def test_luminance_mode_preserves_hue():
    """Scaling by f(Y)/Y must keep channel ratios exactly."""
    rgb = np.array([[2.0, 1.0, 0.5]], dtype=np.float32)
    out = tone_map(rgb, "reinhard", source_peak=PEAK_1000, mode="luminance")
    ratio = out / rgb
    assert np.allclose(ratio, ratio[0, 0], rtol=1e-5)


def test_rgb_mode_desaturates_highlights():
    """Per-channel mode should pull a bright saturated colour toward white."""
    rgb = np.array([[4.0, 1.0, 0.25]], dtype=np.float32)
    lum = tone_map(rgb, "reinhard", source_peak=PEAK_1000, mode="luminance")
    per = tone_map(rgb, "reinhard", source_peak=PEAK_1000, mode="rgb")
    spread = lambda v: float(v.max() - v.min())
    assert spread(per) < spread(lum)


def test_gamut_compress_brings_negatives_into_range():
    rgb = np.array([[1.0, -0.4, -0.2]], dtype=np.float32)
    out = gamut_compress(rgb, "bt709", strength=1.0)
    assert out.min() >= -1e-6, out


def test_gamut_compress_off_is_a_no_op():
    rgb = np.array([[1.0, -0.4, 0.3]], dtype=np.float32)
    assert np.array_equal(gamut_compress(rgb, "bt709", strength=0.0), rgb)


def test_gamut_compress_preserves_neutrals():
    """Greys have no distance from the neutral axis, so nothing may happen."""
    rgb = np.full((4, 3), 0.5, dtype=np.float32)
    assert np.allclose(gamut_compress(rgb, "bt709", strength=1.0), rgb, atol=1e-6)


def test_gamut_compress_is_monotonic_in_strength():
    rgb = np.array([[1.0, -0.5, -0.3]], dtype=np.float32)
    prev = -1.0
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        low = float(gamut_compress(rgb, "bt709", strength=s).min())
        assert low >= prev - 1e-6
        prev = low
