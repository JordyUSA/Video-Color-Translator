"""Primaries, matrices and chromatic adaptation."""

import numpy as np
import pytest

from vct.core.colorimetry import (COLOR_SPACES, get_space, luma_weights,
                                  rgb_to_rgb_matrix, rgb_to_xyz_matrix,
                                  xy_to_XYZ)


@pytest.mark.parametrize("space_id", sorted(COLOR_SPACES))
def test_white_maps_to_white_point(space_id):
    """RGB (1,1,1) must land exactly on the space's white point."""
    space = get_space(space_id)
    xyz = rgb_to_xyz_matrix(space) @ np.ones(3)
    expected = xy_to_XYZ(space.white)
    assert np.allclose(xyz, expected, atol=1e-9)


@pytest.mark.parametrize("space_id", sorted(COLOR_SPACES))
def test_round_trip_through_xyz(space_id):
    m = rgb_to_rgb_matrix(space_id, "bt709") @ rgb_to_rgb_matrix("bt709", space_id)
    assert np.allclose(m, np.eye(3), atol=1e-9)


def test_same_space_is_identity():
    assert np.array_equal(rgb_to_rgb_matrix("bt709", "bt709"), np.eye(3))
    # sRGB is an alias of Rec.709 primaries, so this must be identity too.
    assert np.allclose(rgb_to_rgb_matrix("srgb", "bt709"), np.eye(3), atol=1e-12)


def test_bt709_luma_weights():
    """The familiar 0.2126 / 0.7152 / 0.0722 must fall out of the primaries."""
    assert np.allclose(luma_weights("bt709"), [0.2126, 0.7152, 0.0722], atol=1e-4)


def test_bt2020_luma_weights():
    assert np.allclose(luma_weights("bt2020"), [0.2627, 0.6780, 0.0593], atol=1e-4)


def test_white_point_adaptation_keeps_neutral_neutral():
    """DCI-P3 has a non-D65 white; converting must not tint neutrals."""
    m = rgb_to_rgb_matrix("p3_dci", "bt709")
    out = m @ np.ones(3)
    assert np.allclose(out, out[0], rtol=2e-2), f"neutral drifted: {out}"


def test_wide_gamut_to_709_goes_out_of_range():
    """Saturated wide-gamut colour is genuinely outside Rec.709 - the pipeline
    depends on that being true, so assert it rather than assume it."""
    m = rgb_to_rgb_matrix("bt2020", "bt709")
    assert (m @ np.array([0.0, 1.0, 0.0])).min() < 0.0


def test_unknown_space_raises():
    with pytest.raises(KeyError):
        get_space("rec2021")
