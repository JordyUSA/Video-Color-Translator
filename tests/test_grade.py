"""Primary adjustment behaviour, including the directions the sliders promise."""

import numpy as np
import pytest

from vct.core.grade import (BASE_CCT, GradeParams, black_white_point, cct_to_xy,
                            contrast, exposure, highlights_shadows, saturation,
                            white_balance, white_balance_white_point)

GREY = np.full((1, 3), 0.18, dtype=np.float32)


def test_cct_zero_point_is_d65():
    """The slider's zero must be the display white, or neutral gets a cast."""
    x, y = cct_to_xy(BASE_CCT)
    assert abs(x - 0.3127) < 1e-3 and abs(y - 0.3290) < 1e-3


def test_cct_curve_is_continuous_across_the_locus_join():
    """Planckian below 4000 K, daylight above - the seam must not be visible."""
    a, b = cct_to_xy(3999.0), cct_to_xy(4001.0)
    assert abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 1e-3


def test_exposure_is_stops():
    assert np.allclose(exposure(GREY, 1.0), 0.36, atol=1e-6)
    assert np.allclose(exposure(GREY, -1.0), 0.09, atol=1e-6)
    assert np.array_equal(exposure(GREY, 0.0), GREY)


def test_contrast_holds_the_pivot():
    """Mid grey must not move, or every contrast tweak becomes an exposure tweak."""
    for amount in (-1.0, -0.3, 0.5, 1.0):
        assert np.allclose(contrast(GREY, amount), 0.18, atol=1e-5)


def test_contrast_direction():
    dark = np.full((1, 3), 0.05, dtype=np.float32)
    bright = np.full((1, 3), 0.6, dtype=np.float32)
    assert float(contrast(dark, 0.5)[0, 0]) < 0.05
    assert float(contrast(bright, 0.5)[0, 0]) > 0.6


def test_white_balance_neutral_is_identity():
    assert np.array_equal(white_balance(GREY, 0.0, 0.0), GREY)


def test_white_balance_is_continuous_at_zero():
    """A slider that jumps as it leaves zero is unusable."""
    nudge = white_balance(GREY, 0.001, 0.0)
    assert np.allclose(nudge, GREY, atol=1e-3), nudge


@pytest.mark.parametrize("temp,warmer", [(50.0, True), (-50.0, False)])
def test_temperature_direction(temp, warmer):
    """Positive temperature must warm the image, as the label says."""
    out = white_balance(GREY, temp, 0.0).ravel()
    assert bool(out[0] > out[2]) is warmer


@pytest.mark.parametrize("tint,magenta", [(60.0, True), (-60.0, False)])
def test_tint_direction(tint, magenta):
    """Positive tint is magenta; negative is green (Premiere/Lightroom convention)."""
    r, g, b = white_balance(GREY, 0.0, tint).ravel()
    assert bool((r > g) and (b > g)) is magenta


def test_tint_does_not_change_temperature():
    """Tint moves perpendicular to the locus, so it must not warm or cool."""
    r, _, b = white_balance(GREY, 0.0, 80.0).ravel()
    assert abs(float(r) - float(b)) < 0.02


def test_white_balance_stays_finite_at_the_extremes():
    for temp in (-100.0, 100.0):
        for tint in (-100.0, 100.0):
            out = white_balance(GREY, temp, tint)
            assert np.isfinite(out).all()
            assert out.min() > -0.1, (temp, tint, out)


def test_shadow_lift_does_not_lift_true_black():
    """A gain, not an offset - black must stay black, no milky haze."""
    black = np.zeros((1, 3), dtype=np.float32)
    assert np.allclose(highlights_shadows(black, 0.0, 1.0), 0.0)


def test_highlights_and_shadows_hit_the_right_ends():
    dark = np.full((1, 3), 0.02, dtype=np.float32)
    bright = np.full((1, 3), 0.9, dtype=np.float32)
    # Pulling highlights down must barely touch the shadows and vice versa.
    assert float(highlights_shadows(dark, -1.0, 0.0)[0, 0]) == pytest.approx(0.02, rel=0.05)
    assert float(highlights_shadows(bright, -1.0, 0.0)[0, 0]) < 0.6
    assert float(highlights_shadows(dark, 0.0, 1.0)[0, 0]) > 0.03
    assert float(highlights_shadows(bright, 0.0, 1.0)[0, 0]) == pytest.approx(0.9, rel=0.05)


def test_saturation_extremes():
    rgb = np.array([[0.8, 0.2, 0.4]], dtype=np.float32)
    mono = saturation(rgb, -1.0)
    assert np.allclose(mono, mono[0, 0], atol=1e-6)
    assert np.array_equal(saturation(rgb, 0.0), rgb)
    boosted = saturation(rgb, 0.5)
    assert float(boosted.max() - boosted.min()) > float(rgb.max() - rgb.min())


def test_saturation_leaves_neutrals_alone():
    grey = np.full((3, 3), 0.42, dtype=np.float32)
    assert np.allclose(saturation(grey, 0.8), grey, atol=1e-6)


def test_black_white_point():
    x = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = black_white_point(x, 0.0, 1.0)
    assert np.array_equal(out, x)
    out = black_white_point(x, 0.1, 0.9)
    assert float(out[0, 0]) == pytest.approx(-0.125)
    assert float(out[0, 2]) == pytest.approx(1.125)


def test_grade_params_neutral_and_reset():
    p = GradeParams()
    assert p.is_neutral()
    p.exposure = 1.5
    p.saturation = -0.4
    assert not p.is_neutral()
    p.reset()
    assert p.is_neutral()
