"""Decoding on a worker thread, so the UI never blocks on FFmpeg.

Only decoding happens here.  Colour is applied on the way to the screen by
sampling the LUT, which means a slider change repaints the current frame
without decoding anything again.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, QWaitCondition, Signal

from ..core.pipeline import SourceParams
from ..media.decoder import DEFAULT_PREVIEW_WIDTH, FrameReader
from ..media.ffmpeg import FFmpegTools
from ..media.probe import MediaInfo


class PlaybackThread(QThread):
    """Owns a :class:`FrameReader` and drives it on demand.

    Commands arrive from the UI thread and are picked up at the top of the loop
    rather than acted on immediately, so a burst of scrubbing collapses into one
    seek instead of queueing up a dozen.
    """

    frameReady = Signal(object, float)   # frame (uint16 HxWx3), position seconds
    playbackStopped = Signal()
    endReached = Signal()
    error = Signal(str)

    def __init__(self, info: MediaInfo, source: SourceParams,
                 tools: Optional[FFmpegTools] = None,
                 max_width: int = DEFAULT_PREVIEW_WIDTH, parent=None):
        super().__init__(parent)
        self._reader = FrameReader(info, source, tools, max_width)
        self._info = info

        self._mutex = QMutex()
        self._wake = QWaitCondition()
        self._stopping = False
        self._playing = False
        self._pending_seek: Optional[float] = 0.0
        self._step_requested = False
        self._position = 0.0

    # -- geometry --------------------------------------------------------
    @property
    def preview_size(self) -> tuple:
        return self._reader.width, self._reader.height

    @property
    def position(self) -> float:
        with QMutexLocker(self._mutex):
            return self._position

    @property
    def is_playing(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._playing

    # -- commands --------------------------------------------------------
    def seek(self, seconds: float) -> None:
        with QMutexLocker(self._mutex):
            self._pending_seek = max(0.0, float(seconds))
        self._wake.wakeAll()

    def play(self) -> None:
        with QMutexLocker(self._mutex):
            self._playing = True
        self._wake.wakeAll()

    def pause(self) -> None:
        with QMutexLocker(self._mutex):
            self._playing = False
        self._wake.wakeAll()

    def toggle(self) -> None:
        self.pause() if self.is_playing else self.play()

    def step(self, frames: int = 1) -> None:
        """Advance or retreat by whole frames."""
        rate = self._info.frame_rate or 25.0
        with QMutexLocker(self._mutex):
            self._playing = False
            if frames > 0:
                self._step_requested = True
            else:
                self._pending_seek = max(0.0, self._position + frames / rate)
        self._wake.wakeAll()

    def setSourceParams(self, source: SourceParams) -> None:
        """Interpretation changed: re-decode the current frame under the new one.

        Only the matrix and range live on the FFmpeg side, so this is rare - the
        transfer curve and gamut are handled by the LUT and need no re-decode.
        """
        with QMutexLocker(self._mutex):
            self._reader.source = source
            self._pending_seek = self._position
        self._wake.wakeAll()

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._stopping = True
            self._playing = False
        self._wake.wakeAll()
        self.wait(3000)

    # -- loop ------------------------------------------------------------
    def run(self) -> None:                                # pragma: no cover - thread
        frame_interval = 1.0 / (self._info.frame_rate or 25.0)
        next_due = time.perf_counter()
        try:
            while True:
                with QMutexLocker(self._mutex):
                    if self._stopping:
                        break
                    seek = self._pending_seek
                    self._pending_seek = None
                    playing = self._playing
                    step = self._step_requested
                    self._step_requested = False
                    if not playing and not step and seek is None:
                        self._wake.wait(self._mutex, 200)
                        continue

                if seek is not None:
                    self._reader.open(seek)
                    next_due = time.perf_counter()

                if not self._reader.is_open:
                    self._reader.open(self._position)

                frame = self._reader.read()
                if frame is None:
                    with QMutexLocker(self._mutex):
                        was_playing = self._playing
                        self._playing = False
                    if was_playing or seek is not None:
                        self.endReached.emit()
                        self.playbackStopped.emit()
                    continue

                position = self._reader.position
                with QMutexLocker(self._mutex):
                    self._position = position
                self.frameReady.emit(frame, position)

                if playing:
                    next_due += frame_interval
                    sleep = next_due - time.perf_counter()
                    if sleep > 0:
                        self.msleep(int(sleep * 1000))
                    else:
                        # Behind schedule: give up on catching the missed time
                        # rather than sprinting through frames.
                        next_due = time.perf_counter()
        except Exception as exc:                          # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self._reader.close()
