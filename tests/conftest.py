"""Shared fixtures. Media tests skip cleanly when FFmpeg is unavailable."""

import os
import subprocess

import pytest

from vct.media.ffmpeg import detect_tools

TOOLS = detect_tools()

requires_ffmpeg = pytest.mark.skipif(
    not TOOLS.available or not TOOLS.can_probe,
    reason="ffmpeg/ffprobe not installed",
)


def _lavfi(tools, path, args):
    subprocess.run([tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args, path],
                   check=True, timeout=180)
    return path


@pytest.fixture(scope="session")
def tools():
    return TOOLS


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    """Three synthetic clips covering the cases the tool exists to handle."""
    if not TOOLS.available:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("clips")
    made = {}

    made["hdr10"] = _lavfi(TOOLS, str(out / "hdr10.mp4"), [
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24", "-t", "1",
        "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
        "-x265-params", "log-level=error:max-cll=1000,400",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc", "-tag:v", "hvc1",
    ])
    made["hlg"] = _lavfi(TOOLS, str(out / "hlg.mp4"), [
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25", "-t", "1",
        "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
        "-x265-params", "log-level=error",
        "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
        "-colorspace", "bt2020nc",
    ])
    # 23.976 with audio: the case where framerate preservation actually matters.
    made["sdr"] = _lavfi(TOOLS, str(out / "sdr.mov"), [
        "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24000/1001",
        "-f", "lavfi", "-i", "sine=frequency=440", "-t", "1",
        "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
        "-c:a", "pcm_s16le",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-metadata:s:v", "encoder=Sony XAVC",
    ])
    return made
