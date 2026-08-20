"""Reading frames out of a file, at preview resolution, as RGB48.

Frames come back in the source's *own* encoding - S-Log3 code values stay
S-Log3 code values - because interpreting them is the LUT's job, not the
decoder's.  See :mod:`vct.media.filters`.

Deliberately a plain Python class with no Qt in it, so it can be exercised
headless.  The UI wraps it in a thread.
"""

from __future__ import annotations

import subprocess
import threading
from typing import Iterator, Optional

import numpy as np

from ..core.pipeline import SourceParams
from .ffmpeg import FFmpegError, FFmpegNotFound, FFmpegTools, detect_tools
from .filters import preview_chain
from .probe import MediaInfo

#: Preview width cap. 4K and 6K source decodes far faster scaled down, and the
#: grade is judged on colour rather than on resolution.
DEFAULT_PREVIEW_WIDTH = 1280


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


class FrameReader:
    """Streams RGB frames from one point in a file.

    Seeking restarts the FFmpeg process.  That sounds heavy-handed but at
    preview resolution it takes a few tens of milliseconds, and it avoids the
    whole class of bugs that come from trying to keep a long-lived decoder and a
    UI timeline in agreement.
    """

    def __init__(self, info: MediaInfo, source: SourceParams,
                 tools: Optional[FFmpegTools] = None,
                 max_width: int = DEFAULT_PREVIEW_WIDTH):
        self.info = info
        self.source = source
        self.tools = tools or detect_tools()
        self.max_width = max(160, int(max_width))

        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._position = 0.0
        self._frames_read = 0

        self.width, self.height = self._preview_size()
        self._frame_bytes = self.width * self.height * 3 * 2   # rgb48le

    # -- geometry --------------------------------------------------------
    def _preview_size(self) -> tuple:
        src_w = max(self.info.width, 2)
        src_h = max(self.info.height, 2)
        if src_w <= self.max_width:
            return _even(src_w), _even(src_h)
        width = _even(self.max_width)
        height = _even(max(2, int(round(src_h * width / src_w))))
        return width, height

    # -- process management ----------------------------------------------
    def _build_command(self, start: float) -> list:
        args = [self.tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if start > 0.0:
            # Input seeking: FFmpeg decodes from the preceding keyframe and
            # discards, so this is frame-accurate as well as fast.
            args += ["-ss", f"{start:.6f}"]
        args += [
            "-i", self.info.path,
            "-map", "0:v:0",
            "-vf", preview_chain(self.source, self.width),
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-",
        ]
        return args

    def open(self, start: float = 0.0) -> None:
        """(Re)start decoding from `start` seconds."""
        if not self.tools.available:
            raise FFmpegNotFound("ffmpeg is not available")
        self.close()
        with self._lock:
            self._position = max(0.0, float(start))
            self._frames_read = 0
            self._proc = subprocess.Popen(
                self._build_command(self._position),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_bytes,
            )

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

    @property
    def is_open(self) -> bool:
        return self._proc is not None

    # -- reading ---------------------------------------------------------
    def read(self) -> Optional[np.ndarray]:
        """Next frame as uint16 ``(h, w, 3)``, or None at end of stream.

        uint16 rather than float so the array can go straight into a 16-bit GL
        texture without a conversion pass over every pixel.
        """
        with self._lock:
            proc = self._proc
        if proc is None or proc.stdout is None:
            return None

        raw = proc.stdout.read(self._frame_bytes)
        if not raw or len(raw) < self._frame_bytes:
            self._drain_error(proc)
            return None

        self._frames_read += 1
        frame = np.frombuffer(raw, dtype="<u2").reshape(self.height, self.width, 3)
        return frame

    def _drain_error(self, proc: subprocess.Popen) -> None:
        """Surface a decode failure rather than reporting a silent end of file."""
        if proc.poll() in (None, 0):
            return
        message = ""
        try:
            if proc.stderr:
                message = proc.stderr.read() or b""
                message = message.decode("utf-8", "replace")
        except (OSError, ValueError):
            pass
        if message.strip():
            raise FFmpegError(f"decoding {self.info.path} failed",
                              self._build_command(self._position), message)

    def read_at(self, timestamp: float) -> Optional[np.ndarray]:
        """Seek to `timestamp` seconds and return that frame."""
        self.open(timestamp)
        return self.read()

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    @property
    def position(self) -> float:
        """Playback position in seconds, tracked from frames delivered."""
        rate = self.info.frame_rate or 25.0
        return self._position + self._frames_read / rate

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def to_float(frame: np.ndarray) -> np.ndarray:
    """uint16 preview frame to float32 in 0..1."""
    return frame.astype(np.float32) / 65535.0
