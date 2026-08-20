"""Reading what a file says about itself, via ffprobe."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional

from .ffmpeg import FFmpegError, FFmpegTools, detect_tools

#: pix_fmt name fragments to bit depth. Checked longest-first.
_DEPTH_HINTS = [("16le", 16), ("16be", 16), ("14", 14), ("12", 12), ("10", 10),
                ("9", 9), ("p016", 16), ("p010", 10)]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_rate(value: Any) -> float:
    """FFmpeg rates arrive as 'num/den' strings, sometimes '0/0'."""
    if not value:
        return 0.0
    try:
        frac = Fraction(str(value))
        return float(frac) if frac.denominator else 0.0
    except (ZeroDivisionError, ValueError):
        return 0.0


def _bit_depth(pix_fmt: str, stream: Dict[str, Any]) -> int:
    explicit = stream.get("bits_per_raw_sample")
    if explicit:
        try:
            depth = int(explicit)
            if 8 <= depth <= 16:
                return depth
        except (TypeError, ValueError):
            pass
    name = (pix_fmt or "").lower()
    for token, depth in _DEPTH_HINTS:
        if token in name:
            return depth
    return 8


@dataclass
class MediaInfo:
    """Everything the tool needs to know about an input file."""

    path: str = ""
    container: str = ""
    duration: float = 0.0
    size_bytes: int = 0

    codec: str = ""
    codec_long: str = ""
    profile: str = ""
    width: int = 0
    height: int = 0
    pix_fmt: str = ""
    bit_depth: int = 8
    frame_rate: float = 0.0
    avg_frame_rate: float = 0.0
    frame_count: int = 0
    rotation: int = 0

    # Colour metadata as tagged in the file. Empty string means "not tagged",
    # which is different from "tagged as bt709" and matters for detection.
    color_transfer: str = ""
    color_primaries: str = ""
    color_space: str = ""
    color_range: str = ""

    max_cll: Optional[int] = None
    max_fall: Optional[int] = None
    mastering_peak_nits: Optional[float] = None

    has_audio: bool = False
    audio_codec: str = ""
    stream_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------
    @property
    def resolution(self) -> str:
        return f"{self.width} x {self.height}" if self.width else "unknown"

    @property
    def is_limited_range(self) -> bool:
        """Absent range metadata means limited - that is the video default."""
        return self.color_range.lower() not in ("pc", "full", "jpeg")

    @property
    def range_id(self) -> str:
        return "tv" if self.is_limited_range else "pc"

    @property
    def duration_timecode(self) -> str:
        total = max(self.duration, 0.0)
        hours, rem = divmod(int(total), 3600)
        minutes, seconds = divmod(rem, 60)
        frames = int((total - int(total)) * (self.frame_rate or 25.0))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    @property
    def estimated_frames(self) -> int:
        if self.frame_count:
            return self.frame_count
        if self.duration and self.frame_rate:
            return int(round(self.duration * self.frame_rate))
        return 0

    def summary_rows(self) -> List[tuple]:
        """Label/value pairs for the source info panel."""
        rows = [
            ("File", os.path.basename(self.path)),
            ("Container", self.container or "-"),
            ("Codec", f"{self.codec}{f' ({self.profile})' if self.profile else ''}"),
            ("Resolution", self.resolution),
            ("Frame rate", f"{self.frame_rate:.3f} fps" if self.frame_rate else "-"),
            ("Duration", self.duration_timecode),
            ("Pixel format", f"{self.pix_fmt or '-'} ({self.bit_depth}-bit)"),
            ("Range", "Full (pc)" if not self.is_limited_range else "Limited (tv)"),
            ("Transfer", self.color_transfer or "not tagged"),
            ("Primaries", self.color_primaries or "not tagged"),
            ("Matrix", self.color_space or "not tagged"),
        ]
        if self.max_cll:
            rows.append(("MaxCLL", f"{self.max_cll} cd/m2"))
        if self.mastering_peak_nits:
            rows.append(("Mastering peak", f"{self.mastering_peak_nits:.0f} cd/m2"))
        if self.has_audio:
            rows.append(("Audio", self.audio_codec or "present"))
        return rows


def _parse_side_data(stream: Dict[str, Any], info: MediaInfo) -> None:
    """Pull HDR10 metadata out of the stream's side data blocks."""
    for entry in stream.get("side_data_list", []) or []:
        kind = str(entry.get("side_data_type", "")).lower()
        if "content light" in kind:
            if entry.get("max_content") is not None:
                info.max_cll = int(_to_float(entry.get("max_content")))
            if entry.get("max_average") is not None:
                info.max_fall = int(_to_float(entry.get("max_average")))
        elif "mastering display" in kind:
            # Reported as a rational string like "10000000/10000".
            raw = entry.get("max_luminance")
            if raw is not None:
                try:
                    info.mastering_peak_nits = float(Fraction(str(raw)))
                except (ValueError, ZeroDivisionError):
                    info.mastering_peak_nits = _to_float(raw) or None
        elif "displaymatrix" in kind.replace(" ", ""):
            info.rotation = int(_to_float(entry.get("rotation")))


def _probe_frame_side_data(path: str, tools: FFmpegTools, info: MediaInfo) -> None:
    """Read HDR10 metadata off the first frame.

    HEVC carries mastering display and content light level as SEI messages, so
    they show up in frame side data rather than on the stream.  Reading one frame
    is cheap and it is what lets the source peak field fill itself in instead of
    asking the user to know their own master's nits.
    """
    try:
        proc = tools.run_probe([
            "-loglevel", "error",
            "-select_streams", "v:0",
            "-show_frames", "-read_intervals", "%+#1",
            "-show_entries", "frame=side_data_list",
            "-print_format", "json",
            path,
        ], timeout=20.0)
        frames = json.loads(proc.stdout or "{}").get("frames") or []
    except (FFmpegError, json.JSONDecodeError, OSError):
        return   # Best-effort only; the user can still set the peak by hand.
    if frames:
        _parse_side_data(frames[0], info)


def probe(path: str, tools: Optional[FFmpegTools] = None) -> MediaInfo:
    """Run ffprobe over `path` and return what it says.

    Raises FileNotFoundError if the file is missing, and FFmpegError if ffprobe
    cannot make sense of it (which for this tool most often means an unsupported
    codec).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tools = tools or detect_tools()

    proc = tools.run_probe([
        "-loglevel", "error",
        "-show_streams", "-show_format",
        "-print_format", "json",
        path,
    ])
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"could not parse ffprobe output for {path}",
                          [tools.ffprobe or "ffprobe", path], proc.stderr) from exc

    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"{os.path.basename(path)} has no video stream",
                          [tools.ffprobe or "ffprobe", path], proc.stderr)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {}) or {}

    info = MediaInfo(path=path, raw=data)
    info.container = str(fmt.get("format_name", ""))
    info.duration = _to_float(fmt.get("duration")) or _to_float(video.get("duration"))
    info.size_bytes = int(_to_float(fmt.get("size")))
    info.stream_count = len(streams)
    info.tags = {str(k): str(v) for k, v in (fmt.get("tags") or {}).items()}
    info.tags.update({str(k): str(v) for k, v in (video.get("tags") or {}).items()})

    info.codec = str(video.get("codec_name", ""))
    info.codec_long = str(video.get("codec_long_name", ""))
    info.profile = str(video.get("profile", "") or "")
    info.width = int(_to_float(video.get("width")))
    info.height = int(_to_float(video.get("height")))
    info.pix_fmt = str(video.get("pix_fmt", ""))
    info.bit_depth = _bit_depth(info.pix_fmt, video)
    info.frame_rate = _parse_rate(video.get("r_frame_rate"))
    info.avg_frame_rate = _parse_rate(video.get("avg_frame_rate")) or info.frame_rate
    info.frame_count = int(_to_float(video.get("nb_frames")))

    info.color_transfer = str(video.get("color_transfer", "") or "")
    info.color_primaries = str(video.get("color_primaries", "") or "")
    info.color_space = str(video.get("color_space", "") or "")
    info.color_range = str(video.get("color_range", "") or "")
    _parse_side_data(video, info)

    if audio is not None:
        info.has_audio = True
        info.audio_codec = str(audio.get("codec_name", ""))

    if info.max_cll is None and info.mastering_peak_nits is None:
        _probe_frame_side_data(path, tools, info)
    return info
