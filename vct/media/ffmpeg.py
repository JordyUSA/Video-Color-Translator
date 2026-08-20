"""Locating FFmpeg and finding out what the local build can actually do.

Distro FFmpeg builds differ in which encoders and filters they were compiled
with.  Discovering that once at startup lets the UI grey out what is
unavailable, instead of letting the user configure an export that fails twenty
minutes in.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

#: Checked before PATH, so a user can point at a specific build.
ENV_FFMPEG = "VCT_FFMPEG"
ENV_FFPROBE = "VCT_FFPROBE"

# Keep child processes from opening a console window and from reading stdin.
_RUN_KW = dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
               stderr=subprocess.PIPE, text=True, errors="replace")


class FFmpegNotFound(RuntimeError):
    """Raised when neither a system nor a bundled FFmpeg can be located."""


class FFmpegError(RuntimeError):
    """A child FFmpeg process exited non-zero."""

    def __init__(self, message: str, command: Sequence[str], stderr: str = ""):
        super().__init__(message)
        self.command = list(command)
        self.stderr = stderr

    def __str__(self) -> str:  # pragma: no cover - formatting only
        tail = "\n".join(self.stderr.strip().splitlines()[-8:])
        return f"{super().__str__()}\n{tail}" if tail else super().__str__()


def _bundled_ffmpeg() -> Optional[str]:
    """The static build from the optional imageio-ffmpeg wheel, if installed."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def find_ffmpeg() -> Optional[str]:
    for candidate in (os.environ.get(ENV_FFMPEG), shutil.which("ffmpeg"),
                      _bundled_ffmpeg()):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def find_ffprobe(ffmpeg_path: Optional[str] = None) -> Optional[str]:
    for candidate in (os.environ.get(ENV_FFPROBE), shutil.which("ffprobe")):
        if candidate and os.path.exists(candidate):
            return candidate
    # Some builds ship both binaries side by side even when only one is on PATH.
    if ffmpeg_path:
        sibling = os.path.join(os.path.dirname(ffmpeg_path),
                               "ffprobe" + (".exe" if ffmpeg_path.endswith(".exe") else ""))
        if os.path.exists(sibling):
            return sibling
    return None


@dataclass
class FFmpegTools:
    """Paths and capabilities of the FFmpeg install this session will use."""

    ffmpeg: Optional[str] = None
    ffprobe: Optional[str] = None
    version: str = ""
    filters: Set[str] = field(default_factory=set)
    encoders: Set[str] = field(default_factory=set)

    @property
    def available(self) -> bool:
        return bool(self.ffmpeg)

    @property
    def can_probe(self) -> bool:
        return bool(self.ffprobe)

    def has_filter(self, name: str) -> bool:
        return name in self.filters

    def has_encoder(self, name: str) -> bool:
        return name in self.encoders

    def missing_requirements(self) -> List[str]:
        """Human-readable list of anything the tool needs but cannot find."""
        problems: List[str] = []
        if not self.ffmpeg:
            problems.append("ffmpeg was not found on PATH")
        if not self.ffprobe:
            problems.append("ffprobe was not found on PATH")
        if self.ffmpeg and not self.has_filter("lut3d"):
            problems.append("this ffmpeg build has no 'lut3d' filter, so colour "
                            "transforms cannot be applied on export")
        if self.ffmpeg and not self.has_filter("setparams"):
            problems.append("this ffmpeg build has no 'setparams' filter, so "
                            "colour interpretation overrides cannot be applied")
        return problems

    def run(self, args: Sequence[str], timeout: Optional[float] = 60.0,
            check: bool = True) -> subprocess.CompletedProcess:
        """Run ffmpeg with the given arguments."""
        if not self.ffmpeg:
            raise FFmpegNotFound("ffmpeg is not available")
        cmd = [self.ffmpeg, "-hide_banner", *args]
        proc = subprocess.run(cmd, timeout=timeout, **_RUN_KW)
        if check and proc.returncode != 0:
            raise FFmpegError(f"ffmpeg exited with code {proc.returncode}",
                              cmd, proc.stderr or "")
        return proc

    def run_probe(self, args: Sequence[str], timeout: Optional[float] = 30.0
                  ) -> subprocess.CompletedProcess:
        if not self.ffprobe:
            raise FFmpegNotFound("ffprobe is not available")
        cmd = [self.ffprobe, "-hide_banner", *args]
        proc = subprocess.run(cmd, timeout=timeout, **_RUN_KW)
        if proc.returncode != 0:
            raise FFmpegError(f"ffprobe exited with code {proc.returncode}",
                              cmd, proc.stderr or "")
        return proc


def _list_names(binary: str, flag: str) -> Set[str]:
    """Parse `ffmpeg -filters` / `-encoders` output into a set of names."""
    try:
        proc = subprocess.run([binary, "-hide_banner", flag], timeout=30, **_RUN_KW)
    except (OSError, subprocess.SubprocessError):
        return set()
    names: Set[str] = set()
    for line in (proc.stdout or "").splitlines():
        # Both listings are "<flags> <name> <...>" once past the header.
        match = re.match(r"^\s*[A-Za-z.]{3,6}\s+(\S+)\s", line)
        if match:
            names.add(match.group(1))
    return names


def _version_string(binary: str) -> str:
    try:
        proc = subprocess.run([binary, "-version"], timeout=15, **_RUN_KW)
    except (OSError, subprocess.SubprocessError):
        return ""
    first = (proc.stdout or "").splitlines()
    return first[0].strip() if first else ""


@functools.lru_cache(maxsize=1)
def detect_tools() -> FFmpegTools:
    """Locate FFmpeg and probe its capabilities. Cached for the session."""
    ffmpeg = find_ffmpeg()
    tools = FFmpegTools(ffmpeg=ffmpeg, ffprobe=find_ffprobe(ffmpeg))
    if ffmpeg:
        tools.version = _version_string(ffmpeg)
        tools.filters = _list_names(ffmpeg, "-filters")
        tools.encoders = _list_names(ffmpeg, "-encoders")
    return tools


def reset_detection() -> None:
    """Clear the cache - used by tests and after a user changes the path."""
    detect_tools.cache_clear()
