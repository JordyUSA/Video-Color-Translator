"""LUT baking, .cube I/O, and preview-versus-export agreement."""

import os

import numpy as np
import pytest

from vct.core.lut import (DEFAULT_LUT_SIZE, apply_lut, build_lut, identity_lut,
                          read_cube, write_cube)
from vct.core.pipeline import (ColorPipeline, OutputParams, PipelineParams,
                               SourceParams)


def params(src="rec709", out="rec709", **kw):
    src_kw = {k: v for k, v in kw.items() if k in SourceParams.__dataclass_fields__}
    out_kw = {k: v for k, v in kw.items() if k in OutputParams.__dataclass_fields__}
    return PipelineParams(source=SourceParams(profile_id=src, **src_kw),
                          output=OutputParams(profile_id=out, **out_kw))


def test_identity_lut_axis_order():
    """Red must vary fastest - it is what .cube and glTexImage3D both expect."""
    lut = identity_lut(4)
    flat = lut.reshape(-1, 3)
    assert flat[0].tolist() == [0.0, 0.0, 0.0]
    assert flat[1][0] > flat[1][1]          # second entry stepped red only
    assert flat[-1].tolist() == [1.0, 1.0, 1.0]


def test_neutral_pipeline_bakes_an_identity_lut():
    """The export-side twin of the pipeline identity test."""
    lut = build_lut(params(tone_mapper="clip"), 33)
    assert np.abs(lut - identity_lut(33)).max() < 1e-5


def test_lut_size_is_validated():
    with pytest.raises(ValueError):
        build_lut(params(), 1)
    with pytest.raises(ValueError):
        build_lut(params(), 1000)


def test_cube_round_trip(tmp_path):
    lut = build_lut(params("hdr10", "rec709"), 17)
    path = str(tmp_path / "grade.cube")
    write_cube(path, lut, title="Test LUT")
    back, header = read_cube(path)
    assert header["size"] == 17
    assert header["title"] == "Test LUT"
    assert np.abs(back - lut).max() < 1e-5


def test_cube_file_is_well_formed(tmp_path):
    path = str(tmp_path / "g.cube")
    write_cube(path, build_lut(params(), 5))
    lines = [l for l in open(path).read().splitlines() if l and not l.startswith("#")]
    assert any(l.startswith("LUT_3D_SIZE 5") for l in lines)
    data = [l for l in lines if not l.split()[0][0].isalpha()]
    assert len(data) == 125


def test_read_cube_rejects_a_1d_lut(tmp_path):
    path = tmp_path / "one.cube"
    path.write_text("LUT_1D_SIZE 4\n0 0 0\n1 1 1\n")
    with pytest.raises(ValueError, match="1D LUT"):
        read_cube(str(path))


def test_read_cube_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "short.cube"
    path.write_text("LUT_3D_SIZE 4\n" + "0.5 0.5 0.5\n" * 10)
    with pytest.raises(ValueError, match="expected"):
        read_cube(str(path))


def test_apply_identity_lut_changes_nothing():
    img = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    assert np.allclose(apply_lut(img, identity_lut(33)), img, atol=1e-6)


def test_apply_lut_preserves_shape():
    img = np.zeros((7, 5, 3), dtype=np.float32)
    assert apply_lut(img, identity_lut(9)).shape == (7, 5, 3)


def test_apply_lut_clamps_out_of_range_input():
    img = np.array([[[-0.5, 1.5, 0.5]]], dtype=np.float32)
    out = apply_lut(img, identity_lut(17))
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("src", ["hdr10", "hlg", "slog3_sgamut3cine", "clog3_cinemagamut"])
def test_lut_matches_the_pipeline_it_was_baked_from(src):
    """The core promise: the preview LUT and the export LUT are one transform.

    Tolerance is set from what a 33-cube can represent. Errors concentrate on
    extreme out-of-gamut colours that real footage rarely contains, so the mean
    matters more than the peak here - see docs for raising the LUT size.
    """
    p = params(src, "rec709", source_peak_nits=1000.0)
    probe = np.random.default_rng(3).random((5000, 3)).astype(np.float32)
    direct = ColorPipeline(p).transform(probe)
    via_lut = apply_lut(probe, build_lut(p, DEFAULT_LUT_SIZE))
    err = np.abs(direct - via_lut)
    assert err.mean() < 1.5 / 255.0, f"{src} mean error {err.mean() * 255:.2f}/255"
    assert np.percentile(err, 99.0) < 8.0 / 255.0
    assert err.max() < 24.0 / 255.0


def test_bigger_luts_are_more_accurate():
    """Raising the size must actually buy precision, or the option is a lie."""
    p = params("slog3_sgamut3cine", "rec709")
    probe = np.random.default_rng(5).random((4000, 3)).astype(np.float32)
    direct = ColorPipeline(p).transform(probe)
    coarse = np.abs(direct - apply_lut(probe, build_lut(p, 17))).mean()
    fine = np.abs(direct - apply_lut(probe, build_lut(p, 65))).mean()
    assert fine < coarse / 2.0


def test_lut_build_is_fast_enough_to_drag_a_slider():
    """The whole architecture rests on this being milliseconds, not seconds."""
    import time
    p = params("hdr10", "rec709")
    build_lut(p, DEFAULT_LUT_SIZE)          # warm up numpy
    start = time.perf_counter()
    for _ in range(5):
        build_lut(p, DEFAULT_LUT_SIZE)
    elapsed = (time.perf_counter() - start) / 5.0
    assert elapsed < 0.060, f"LUT rebuild took {elapsed * 1000:.1f} ms"
