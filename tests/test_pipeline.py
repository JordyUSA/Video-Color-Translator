"""End-to-end colour pipeline behaviour."""

import numpy as np
import pytest

from vct.core.camera_profiles import PROFILES, get_output_profile, get_profile
from vct.core.pipeline import (ColorPipeline, OutputParams, PipelineParams,
                               SourceParams, signal_to_curve_domain)


def make(src="rec709", out="rec709", **kw):
    src_kw = {k: v for k, v in kw.items() if k in SourceParams.__dataclass_fields__}
    out_kw = {k: v for k, v in kw.items() if k in OutputParams.__dataclass_fields__}
    return PipelineParams(source=SourceParams(profile_id=src, **src_kw),
                          output=OutputParams(profile_id=out, **out_kw))


RAMP = np.linspace(0.0, 1.0, 64, dtype=np.float32)[:, None].repeat(3, axis=1)


def test_neutral_conversion_is_the_identity():
    """Rec.709 in, Rec.709 out, no adjustments: the picture must not change.

    This is the single most important guard in the suite. Any accidental extra
    transfer decode, gamut matrix or range expansion shows up here immediately.
    """
    out = ColorPipeline(make(tone_mapper="clip")).transform(RAMP)
    assert np.allclose(out, RAMP, atol=1e-5), \
        f"max drift {np.abs(out - RAMP).max():.2e}"


def test_bypass_returns_the_source_untouched():
    p = make("hdr10", "rec709")
    p.bypass = True
    assert np.allclose(ColorPipeline(p).transform(RAMP), RAMP, atol=1e-6)


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_every_profile_produces_a_valid_image(profile_id):
    """No profile may emit NaN, or blow past the display range."""
    probe = np.random.default_rng(7).random((512, 3)).astype(np.float32)
    out = ColorPipeline(make(profile_id, "rec709")).transform(probe)
    assert np.isfinite(out).all(), f"{profile_id} produced non-finite output"
    assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_every_profile_is_monotonic_on_a_neutral_ramp(profile_id):
    """A grey ramp must stay a grey ramp - no inversions, no banding kinks."""
    out = ColorPipeline(make(profile_id, "rec709")).transform(RAMP)
    assert np.all(np.diff(out[:, 0]) >= -1e-4), profile_id


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_neutral_input_stays_neutral(profile_id):
    """Grey in, grey out - a colour cast here means a broken matrix."""
    out = ColorPipeline(make(profile_id, "rec709")).transform(RAMP)
    assert np.allclose(out, out[:, :1], atol=2e-3), profile_id


def test_log_source_lands_mid_grey_in_a_sane_place():
    """S-Log3 mid grey should come out somewhere around 40-60% on a Rec.709
    display: dark enough to have highlight headroom, bright enough to be a face."""
    grey = np.full((1, 3), 420.0 / 1023.0, dtype=np.float32)
    p = make("slog3_sgamut3cine", "rec709", source_range="pc")
    out = float(ColorPipeline(p).transform(grey)[0, 0])
    assert 0.35 < out < 0.65, f"S-Log3 grey landed at {out:.3f}"


def test_hdr_highlights_survive_instead_of_clipping():
    """A PQ highlight well above diffuse white must remain distinguishable from
    diffuse white - that is exactly what tone mapping is for."""
    from vct.core.transfer import pq_encode
    white = np.full((1, 3), float(pq_encode(np.array(203.0 / 10000.0))), np.float32)
    bright = np.full((1, 3), float(pq_encode(np.array(1000.0 / 10000.0))), np.float32)
    p = make("hdr10", "rec709", source_peak_nits=1000.0, tone_mapper="bt2390")
    pipe = ColorPipeline(p)
    w = float(pipe.transform(white)[0, 0])
    b = float(pipe.transform(bright)[0, 0])
    assert b > w + 0.02, f"highlight {b:.3f} collapsed onto white {w:.3f}"
    assert w > 0.5, f"diffuse white crushed to {w:.3f}"


def test_clip_operator_does_blow_out_the_same_highlight():
    """The contrast case: without tone mapping that highlight is simply gone."""
    from vct.core.transfer import pq_encode
    white = np.full((1, 3), float(pq_encode(np.array(203.0 / 10000.0))), np.float32)
    bright = np.full((1, 3), float(pq_encode(np.array(1000.0 / 10000.0))), np.float32)
    p = make("hdr10", "rec709", source_peak_nits=1000.0, tone_mapper="clip")
    pipe = ColorPipeline(p)
    assert float(pipe.transform(white)[0, 0]) == pytest.approx(1.0, abs=1e-3)
    assert float(pipe.transform(bright)[0, 0]) == pytest.approx(1.0, abs=1e-3)


def test_limited_range_expansion_for_code_referenced_curves():
    """S-Log3 black sits at code 95; a tv-range decode hands us (95-64)/876."""
    signal = np.array([(95.0 - 64.0) / 876.0], dtype=np.float32)
    code = signal_to_curve_domain(signal, True, "tv", 10)
    assert float(code[0]) == pytest.approx(95.0 / 1023.0, abs=1e-4)


def test_full_range_needs_no_expansion():
    signal = np.array([0.5], dtype=np.float32)
    assert np.array_equal(signal_to_curve_domain(signal, True, "pc", 10), signal)


def test_signal_curves_are_never_expanded():
    """PQ and Rec.709 are defined on the expanded signal, so they must be left be."""
    signal = np.array([0.5], dtype=np.float32)
    assert np.array_equal(signal_to_curve_domain(signal, False, "tv", 10), signal)


def test_range_misinterpretation_actually_matters():
    """If this were a no-op difference, the source-range control would be theatre.

    The cost shows up in the shadows, where the LOG curve is steepest: reading
    S-Log3 in the wrong range moves near-black by most of a stop. Mid grey barely
    budges, which is exactly why the mistake is easy to miss on a bright frame.
    """
    shadow = np.full((1, 3), 0.12, dtype=np.float32)
    tv = ColorPipeline(make("slog3_sgamut3cine", source_range="tv")).transform(shadow)
    pc = ColorPipeline(make("slog3_sgamut3cine", source_range="pc")).transform(shadow)
    assert abs(float(tv[0, 0]) - float(pc[0, 0])) > 0.02


def test_gamut_compression_auto_switches_on_only_when_needed():
    assert ColorPipeline(make("rec709", "rec709")).gamut_compression_amount() == 0.0
    assert ColorPipeline(make("slog3_sgamut3cine", "rec709")).gamut_compression_amount() == 1.0
    p = make("slog3_sgamut3cine", "rec709", gamut_compression=0.0)
    assert ColorPipeline(p).gamut_compression_amount() == 0.0


def test_grade_controls_move_the_image_the_right_way():
    p = make(tone_mapper="clip")
    base = ColorPipeline(p).transform(RAMP)
    p.grade.exposure = 1.0
    assert ColorPipeline(p).transform(RAMP)[32, 0] > base[32, 0]
    p.grade.exposure = 0.0
    p.grade.saturation = -1.0
    mono = ColorPipeline(p).transform(np.array([[0.8, 0.2, 0.4]], np.float32))
    assert np.allclose(mono, mono[0, 0], atol=1e-5)


def test_source_and_output_overrides_beat_the_profile():
    p = make("rec709", "rec709")
    p.source.transfer = "slog3"
    p.source.primaries = "s_gamut3_cine"
    assert p.source.resolved_transfer() == "slog3"
    assert p.source.resolved_primaries() == "s_gamut3_cine"


def test_params_copy_is_deep_enough_to_be_safe():
    p = make()
    q = p.copy()
    q.grade.exposure = 2.0
    q.source.profile_id = "hdr10"
    assert p.grade.exposure == 0.0
    assert p.source.profile_id == "rec709"


@pytest.mark.parametrize("out_id", sorted(get_output_profile.__globals__["OUTPUT_PROFILES"]))
def test_output_profiles_all_work(out_id):
    out = ColorPipeline(make("hdr10", out_id)).transform(RAMP)
    assert np.isfinite(out).all()


def test_profile_registry_is_internally_consistent():
    """Every profile must name a transfer and gamut that actually exist."""
    from vct.core.colorimetry import get_space
    from vct.core.transfer import get_transfer
    for profile in PROFILES.values():
        get_transfer(profile.transfer)
        get_space(profile.primaries)
        assert profile.group in ("Standard", "HDR", "Camera LOG")
