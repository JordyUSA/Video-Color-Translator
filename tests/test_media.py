"""FFmpeg discovery, probing, detection and filter construction."""

import numpy as np
import pytest

from vct.core.pipeline import SourceParams
from vct.media import filters
from vct.media.detect import (GUESS, TAGGED, UNKNOWN, detect, normalise_primaries,
                              normalise_transfer, suggested_peak_nits)
from vct.media.ffmpeg import FFmpegTools, detect_tools
from vct.media.probe import MediaInfo, probe

from .conftest import requires_ffmpeg


# ---------------------------------------------------------------- discovery

def test_tools_report_what_is_missing():
    empty = FFmpegTools()
    problems = empty.missing_requirements()
    assert not empty.available
    assert any("ffmpeg" in p for p in problems)


def test_tools_flag_a_build_without_lut3d():
    """A build without lut3d cannot export, and must say so up front."""
    crippled = FFmpegTools(ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe",
                           filters={"scale"}, encoders={"libx264"})
    assert any("lut3d" in p for p in crippled.missing_requirements())


@requires_ffmpeg
def test_real_build_has_what_we_need():
    tools = detect_tools()
    assert tools.available and tools.can_probe
    assert tools.has_filter("lut3d") and tools.has_filter("setparams")
    assert tools.encoders, "no encoders parsed out of ffmpeg -encoders"


# ------------------------------------------------------------------ probing

@requires_ffmpeg
def test_probe_reads_hdr10_metadata(clips):
    info = probe(clips["hdr10"])
    assert info.width == 320 and info.height == 180
    assert info.bit_depth == 10
    assert info.color_transfer == "smpte2084"
    assert info.color_primaries == "bt2020"
    assert info.frame_rate == pytest.approx(24.0)
    assert info.is_limited_range and info.range_id == "tv"
    # From the SEI on the first frame, not the stream header.
    assert info.max_cll == 1000


@requires_ffmpeg
def test_probe_reads_fractional_framerate_exactly(clips):
    info = probe(clips["sdr"])
    assert info.frame_rate == pytest.approx(24000 / 1001, abs=1e-6)
    assert info.has_audio


@requires_ffmpeg
def test_probe_rejects_a_missing_file():
    with pytest.raises(FileNotFoundError):
        probe("/nonexistent/file.mov")


@requires_ffmpeg
def test_probe_rejects_a_non_video_file(tmp_path):
    from vct.media.ffmpeg import FFmpegError
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a video")
    with pytest.raises(FFmpegError):
        probe(str(junk))


def test_media_info_derived_fields():
    info = MediaInfo(path="/tmp/a.mov", width=1920, height=1080,
                     frame_rate=25.0, duration=10.48, color_range="")
    assert info.resolution == "1920 x 1080"
    assert info.is_limited_range          # absent range metadata means limited
    assert info.duration_timecode == "00:00:10:12"
    assert info.estimated_frames == 262
    assert any(label == "Transfer" for label, _ in info.summary_rows())


def test_full_range_is_recognised():
    assert not MediaInfo(color_range="pc").is_limited_range
    assert MediaInfo(color_range="tv").is_limited_range


# ---------------------------------------------------------------- detection

def test_pq_is_detected_from_metadata_and_trusted():
    info = MediaInfo(color_transfer="smpte2084", color_primaries="bt2020")
    result = detect(info)
    assert result.profile_id == "hdr10"
    assert result.confidence == TAGGED and result.is_trustworthy


def test_hlg_is_detected_from_metadata():
    result = detect(MediaInfo(color_transfer="arib-std-b67", color_primaries="bt2020"))
    assert result.profile_id == "hlg" and result.confidence == TAGGED


def test_log_is_only_ever_a_guess():
    """The heart of the honesty requirement: S-Log3 footage is tagged bt709, so
    a camera-name hint must never be presented as a detection."""
    info = MediaInfo(color_transfer="bt709", color_primaries="bt709",
                     tags={"encoder": "Sony XAVC"})
    result = detect(info)
    assert result.profile_id == "slog3_sgamut3cine"
    assert result.confidence == GUESS
    assert not result.is_trustworthy
    assert "nothing in the file confirms" in result.reason


def test_untagged_file_defaults_to_709_and_says_so():
    result = detect(MediaInfo())
    assert result.profile_id == "rec709"
    assert result.confidence == UNKNOWN
    assert not result.is_trustworthy


def test_plain_rec709_is_detected_without_a_camera_hint():
    result = detect(MediaInfo(color_transfer="bt709", color_primaries="bt709",
                              color_space="bt709"))
    assert result.profile_id == "rec709" and result.confidence == TAGGED


def test_peak_nits_prefers_declared_metadata():
    assert suggested_peak_nits(MediaInfo(mastering_peak_nits=4000.0)) == 4000.0
    assert suggested_peak_nits(MediaInfo(max_cll=600)) == 600.0
    assert suggested_peak_nits(MediaInfo(color_transfer="arib-std-b67")) == 1000.0
    assert suggested_peak_nits(MediaInfo(), default=203.0) == 203.0


def test_transfer_and_primaries_normalisation():
    assert normalise_transfer("smpte2084") == "pq"
    assert normalise_transfer("ARIB-STD-B67") == "hlg"
    assert normalise_transfer("nonsense") == ""
    assert normalise_primaries("bt2020") == "bt2020"
    assert normalise_primaries("smpte432") == "p3_d65"


@requires_ffmpeg
def test_detection_on_real_files(clips):
    assert detect(probe(clips["hdr10"])).profile_id == "hdr10"
    assert detect(probe(clips["hlg"])).profile_id == "hlg"
    slog = detect(probe(clips["sdr"]))
    assert slog.confidence == GUESS   # the Sony tag is a hint, nothing more


# ------------------------------------------------------------------ filters

def test_setparams_encodes_the_interpretation():
    chain = filters.setparams_filter(SourceParams(profile_id="hdr10"))
    assert "color_trc=smpte2084" in chain
    assert "color_primaries=bt2020" in chain
    assert "colorspace=bt2020nc" in chain
    assert "range=tv" in chain


def test_matrix_override_wins():
    src = SourceParams(profile_id="hdr10", matrix="bt709")
    assert filters.source_matrix_name(src) == "bt709"


def test_scale_is_told_the_matrix_explicitly():
    """Relying on metadata inheritance has broken between FFmpeg releases."""
    chain = filters.preview_chain(SourceParams(profile_id="hdr10"), 1280)
    assert "in_color_matrix=bt2020nc" in chain
    assert "in_range=limited" in chain and "out_range=full" in chain


def test_full_range_source_is_passed_through_as_full():
    chain = filters.preview_chain(SourceParams(profile_id="rec709", source_range="pc"), 640)
    assert "in_range=full" in chain


def test_export_chain_applies_the_lut_before_resizing():
    """Resizing first would blend colours that have not been converted yet."""
    chain = filters.export_chain(SourceParams(), "/tmp/g.cube", out_size=(1920, 1080))
    assert chain.index("lut3d") < chain.index("1920:1080")


def test_filter_paths_are_escaped():
    chain = filters.export_chain(SourceParams(), "/tmp/my lut:v2.cube")
    assert "v2.cube" in chain and "\\:" in chain


def test_output_tagging():
    args = filters.output_tagging_args("rec709")
    assert args[args.index("-color_trc") + 1] == "bt709"
    assert args[args.index("-color_range") + 1] == "tv"
