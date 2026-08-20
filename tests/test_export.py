"""Decode and export, run against real FFmpeg on real files.

These are the tests that check the architecture actually holds: that the LUT
the preview samples and the LUT FFmpeg applies produce the same picture, and
that an export preserves everything the user did not ask to change.
"""

import os

import numpy as np
import pytest

from vct.core.lut import apply_lut, build_lut, write_cube
from vct.core.pipeline import (ColorPipeline, OutputParams, PipelineParams,
                               SourceParams)
from vct.media.decoder import FrameReader, to_float
from vct.media.exporter import (CODECS, ExportJob, ExportSettings,
                                default_output_path, quality_to_crf)
from vct.media.filters import export_chain, preview_chain
from vct.media.probe import probe

from .conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def hdr_params(peak=1000.0):
    return PipelineParams(
        source=SourceParams(profile_id="hdr10", source_peak_nits=peak, bit_depth=10),
        output=OutputParams(profile_id="rec709"))


# ------------------------------------------------------------------ decoding

def test_decoder_reads_every_frame(clips, tools):
    info = probe(clips["hdr10"])
    with FrameReader(info, SourceParams(profile_id="hdr10"), tools) as reader:
        reader.open()
        frames = sum(1 for _ in reader.frames())
    assert frames == info.estimated_frames == 24


def test_decoder_scales_down_large_sources(clips, tools):
    info = probe(clips["hdr10"])
    reader = FrameReader(info, SourceParams(), tools, max_width=160)
    assert reader.width == 160 and reader.height == 90


def test_decoder_does_not_upscale_small_sources(clips, tools):
    info = probe(clips["hdr10"])
    reader = FrameReader(info, SourceParams(), tools, max_width=4096)
    assert (reader.width, reader.height) == (320, 180)


def test_decoder_frames_are_16_bit_rgb(clips, tools):
    info = probe(clips["hdr10"])
    with FrameReader(info, SourceParams(profile_id="hdr10"), tools) as reader:
        frame = reader.read_at(0.0)
    assert frame is not None
    assert frame.dtype == np.uint16
    assert frame.shape == (reader.height, reader.width, 3)
    assert 0.0 <= to_float(frame).max() <= 1.0


def test_decoder_seeking_lands_somewhere_different(clips, tools):
    info = probe(clips["hdr10"])
    with FrameReader(info, SourceParams(profile_id="hdr10"), tools) as reader:
        first = reader.read_at(0.0)
        later = reader.read_at(0.7)
    assert first is not None and later is not None
    assert not np.array_equal(first, later), "seek returned the same frame"


def test_decoder_reports_position(clips, tools):
    info = probe(clips["sdr"])
    with FrameReader(info, SourceParams(), tools) as reader:
        reader.open(0.5)
        reader.read()
        assert 0.5 <= reader.position <= 0.6


# -------------------------------------------------- preview / export parity

def _raw_rgb(tools, path, vf, width, height, frames=2):
    """Run a filter chain and return its output as float RGB, codec-free."""
    import subprocess
    cmd = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", path, "-map", "0:v:0", "-vf", vf, "-frames:v", str(frames),
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    data = np.frombuffer(out, "<u2").reshape(frames, height, width, 3)
    return data.astype(np.float32) / 65535.0


@pytest.mark.parametrize("lut_size,tolerance", [(33, 0.15), (65, 0.05)])
def test_preview_and_export_agree(clips, tools, tmp_path, lut_size, tolerance):
    """The central claim of the design.

    The preview samples the LUT with trilinear interpolation on the GPU; FFmpeg
    samples the same LUT tetrahedrally. If these two ever diverged, every export
    would be a surprise. Compared on real decoded HDR frames.
    """
    path = clips["hdr10"]
    info = probe(path)
    params = hdr_params()

    source = _raw_rgb(tools, path, preview_chain(params.source, info.width),
                      info.width, info.height)

    lut = build_lut(params, lut_size)
    lut_path = str(tmp_path / f"g{lut_size}.cube")
    write_cube(lut_path, lut)
    exported = _raw_rgb(tools, path,
                        export_chain(params.source, lut_path) + ",format=rgb48le",
                        info.width, info.height)

    preview = apply_lut(source, lut)
    error = np.abs(exported - preview)
    assert error.mean() < 0.5 / 255.0, f"mean {error.mean() * 255:.3f}/255"
    assert np.percentile(error, 99.9) < 4.0 / 255.0
    assert error.max() < tolerance


def test_export_matches_the_cpu_pipeline(clips, tools, tmp_path):
    """And both match the pipeline they were baked from, not just each other."""
    path = clips["hdr10"]
    info = probe(path)
    params = hdr_params()
    source = _raw_rgb(tools, path, preview_chain(params.source, info.width),
                      info.width, info.height)
    lut_path = str(tmp_path / "g.cube")
    write_cube(lut_path, build_lut(params, 65))
    exported = _raw_rgb(tools, path,
                        export_chain(params.source, lut_path) + ",format=rgb48le",
                        info.width, info.height)
    direct = ColorPipeline(params).transform(source)
    assert np.abs(exported - direct).mean() < 0.5 / 255.0


# -------------------------------------------------------------------- export

def test_export_preserves_framerate_and_frame_count(clips, tmp_path):
    """23.976 must come out as 23.976, with every frame intact."""
    info = probe(clips["sdr"])
    out = str(tmp_path / "graded.mp4")
    job = ExportJob(info, PipelineParams(
        source=SourceParams(profile_id="slog3_sgamut3cine"),
        output=OutputParams(profile_id="rec709")),
        ExportSettings(output_path=out, codec="h264", container="mp4"))
    assert job.validate() == []
    result = job.run()
    assert result.ok, result.message

    rendered = probe(out)
    assert rendered.frame_rate == pytest.approx(info.frame_rate, abs=1e-4)
    assert rendered.estimated_frames == info.estimated_frames
    assert rendered.duration == pytest.approx(info.duration, abs=0.05)
    assert (rendered.width, rendered.height) == (info.width, info.height)


def test_export_tags_the_output_colour_space(clips, tmp_path):
    """An untagged or mistagged export gets converted again by the player."""
    info = probe(clips["hdr10"])
    out = str(tmp_path / "tagged.mp4")
    ExportJob(info, hdr_params(),
              ExportSettings(output_path=out, codec="h264")).run()
    rendered = probe(out)
    assert rendered.color_transfer == "bt709"
    assert rendered.color_primaries == "bt709"


def test_export_carries_audio_through(clips, tmp_path):
    info = probe(clips["sdr"])
    assert info.has_audio
    out = str(tmp_path / "with_audio.mov")
    result = ExportJob(info, PipelineParams(), ExportSettings(
        output_path=out, codec="prores", container="mov", audio_mode="copy")).run()
    assert result.ok, result.message
    assert probe(out).has_audio


def test_export_can_drop_audio(clips, tmp_path):
    info = probe(clips["sdr"])
    out = str(tmp_path / "silent.mp4")
    ExportJob(info, PipelineParams(), ExportSettings(
        output_path=out, codec="h264", audio_mode="none")).run()
    assert not probe(out).has_audio


def test_export_reports_progress(clips, tmp_path):
    info = probe(clips["hdr10"])
    seen = []
    ExportJob(info, hdr_params(), ExportSettings(
        output_path=str(tmp_path / "p.mp4"), codec="h264")).run(seen.append)
    assert seen, "no progress was reported"
    assert seen[-1].fraction == pytest.approx(1.0, abs=0.05)
    assert seen[-1].frame == info.estimated_frames


def test_export_can_resize(clips, tmp_path):
    info = probe(clips["hdr10"])
    out = str(tmp_path / "small.mp4")
    ExportJob(info, hdr_params(), ExportSettings(
        output_path=out, codec="h264", width=160, height=90)).run()
    rendered = probe(out)
    assert (rendered.width, rendered.height) == (160, 90)


def test_export_actually_tone_maps(clips, tools, tmp_path):
    """A real HDR export must differ from simply relabelling the file."""
    info = probe(clips["hdr10"])
    out = str(tmp_path / "toned.mov")
    ExportJob(info, hdr_params(), ExportSettings(
        output_path=out, codec="prores", container="mov")).run()

    graded = _raw_rgb(tools, out, "scale=in_range=limited:out_range=full,format=rgb48le",
                      info.width, info.height, frames=1)
    untouched = _raw_rgb(tools, clips["hdr10"],
                         preview_chain(SourceParams(profile_id="hdr10"), info.width),
                         info.width, info.height, frames=1)
    assert np.abs(graded - untouched).mean() > 0.02, "export looks unprocessed"


def test_tone_mapping_recovers_highlights_through_the_real_export(clips, tools, tmp_path):
    """Tone mapped versus clipped, rendered by FFmpeg and compared in the same
    domain: the roll-off must leave fewer pixels pinned at maximum.

    Comparing the export against the PQ source directly would prove nothing -
    those are two different encodings, and "how many pixels sit at 1.0" does not
    mean the same thing in each.
    """
    info = probe(clips["hdr10"])
    rendered = {}
    for mapper in ("bt2390", "clip"):
        params = hdr_params()
        params.output.tone_mapper = mapper
        out = str(tmp_path / f"tm_{mapper}.mov")
        result = ExportJob(info, params, ExportSettings(
            output_path=out, codec="prores", container="mov")).run()
        assert result.ok, result.message
        rendered[mapper] = _raw_rgb(
            tools, out, "scale=in_range=limited:out_range=full,format=rgb48le",
            info.width, info.height, frames=1)

    clipped = (rendered["clip"] > 0.999).mean()
    toned = (rendered["bt2390"] > 0.999).mean()
    assert toned < clipped, f"tone mapped {toned:.3f} clipped no less than {clipped:.3f}"


def test_prores_profiles_pick_the_right_pixel_format():
    assert ExportSettings(codec="prores", prores_profile=3).resolved_pix_fmt() == "yuv422p10le"
    assert ExportSettings(codec="prores", prores_profile=4).resolved_pix_fmt() == "yuva444p10le"


def test_quality_maps_to_sensible_crf():
    assert quality_to_crf(100) < quality_to_crf(50) < quality_to_crf(0)
    assert 0 <= quality_to_crf(100) <= 51
    assert 0 <= quality_to_crf(0) <= 51


def test_validation_catches_bad_combinations(clips):
    info = probe(clips["sdr"])
    # ProRes cannot go in an mp4.
    job = ExportJob(info, PipelineParams(), ExportSettings(
        output_path="/tmp/x.mp4", codec="prores", container="mp4"))
    assert any("mp4" in p for p in job.validate())
    # Nor may an export overwrite its own source.
    job = ExportJob(info, PipelineParams(), ExportSettings(
        output_path=info.path, codec="h264"))
    assert any("overwrite the source" in p for p in job.validate())
    # Nor run with no destination.
    assert ExportJob(info, PipelineParams(), ExportSettings()).validate()


def test_default_output_path_never_overwrites(tmp_path):
    source = tmp_path / "clip.mov"
    source.write_text("x")
    first = default_output_path(str(source), "mp4")
    assert first.endswith("clip_graded.mp4")
    open(first, "w").close()
    assert default_output_path(str(source), "mp4").endswith("clip_graded_2.mp4")


def test_every_codec_can_actually_encode(clips, tmp_path, tools):
    """Guards against a codec option the local build cannot honour."""
    info = probe(clips["sdr"])
    for codec_id, option in CODECS.items():
        if not tools.has_encoder(option.encoder):
            pytest.skip(f"{option.encoder} not in this build")
        container = option.containers[0]
        out = str(tmp_path / f"out_{codec_id}.{container}")
        result = ExportJob(info, PipelineParams(), ExportSettings(
            output_path=out, codec=codec_id, container=container)).run()
        assert result.ok, f"{codec_id}: {result.message}"
        assert os.path.getsize(out) > 0
