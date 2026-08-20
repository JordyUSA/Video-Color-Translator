"""Rendering a graded file, by handing the baked LUT to FFmpeg.

The colour transform is applied by ``lut3d`` using the exact table the preview
is showing, so the render matches the screen and runs at encoder speed rather
than at the speed of Python.

Defaults are chosen to preserve everything the user did not ask to change:
source frame rate and timestamps, source resolution, audio streams, and
timecode.  "Match source" is the answer to every question the export dialog
asks unless it is told otherwise.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..core.lut import DEFAULT_LUT_SIZE, build_lut, write_cube
from ..core.pipeline import PipelineParams
from .ffmpeg import FFmpegError, FFmpegNotFound, FFmpegTools, detect_tools
from .filters import export_chain, output_tagging_args
from .probe import MediaInfo


@dataclass(frozen=True)
class CodecOption:
    id: str
    label: str
    encoder: str
    pix_fmts: Tuple[str, ...]
    containers: Tuple[str, ...]
    #: "crf" codecs take a quality number; "profile" codecs take a named tier.
    rate_control: str
    note: str = ""


CODECS: Dict[str, CodecOption] = {c.id: c for c in [
    CodecOption("h264", "H.264 (libx264)", "libx264",
                ("yuv420p", "yuv422p10le"), ("mp4", "mov", "mkv"), "crf",
                "Universally playable. 8-bit 4:2:0 by default."),
    CodecOption("h265", "H.265 / HEVC (libx265)", "libx265",
                ("yuv420p10le", "yuv420p"), ("mp4", "mov", "mkv"), "crf",
                "Better quality per bit; 10-bit by default to avoid banding."),
    CodecOption("prores", "Apple ProRes (prores_ks)", "prores_ks",
                ("yuv422p10le", "yuva444p10le"), ("mov", "mkv"), "profile",
                "Edit-friendly intermediate. Large files, minimal generation loss."),
]}

#: ProRes profile number -> label. The numbers are prores_ks's -profile:v values.
PRORES_PROFILES: Dict[int, str] = {
    0: "Proxy", 1: "LT", 2: "422 (Standard)", 3: "422 HQ", 4: "4444", 5: "4444 XQ",
}

CONTAINER_EXTENSIONS = {"mp4": ".mp4", "mov": ".mov", "mkv": ".mkv"}

#: Quality slider (0-100) to CRF. Low CRF is high quality.
_CRF_WORST, _CRF_BEST = 32, 10


def quality_to_crf(quality: float) -> int:
    q = max(0.0, min(100.0, float(quality)))
    return int(round(_CRF_WORST - (q / 100.0) * (_CRF_WORST - _CRF_BEST)))


@dataclass
class ExportSettings:
    """Everything the export dialog collects."""

    output_path: str = ""
    codec: str = "h265"
    container: str = "mp4"
    quality: float = 70.0              # 0-100, mapped to CRF
    preset: str = "medium"
    prores_profile: int = 3
    pix_fmt: Optional[str] = None      # None: the codec's default
    bitrate_kbps: Optional[int] = None  # set to use bitrate instead of CRF
    width: Optional[int] = None        # None: match source
    height: Optional[int] = None
    fps: Optional[float] = None        # None: match source, timestamps untouched
    audio_mode: str = "copy"           # "copy" | "aac" | "none"
    lut_size: int = DEFAULT_LUT_SIZE
    lut_interp: str = "tetrahedral"
    overwrite: bool = True

    def codec_option(self) -> CodecOption:
        return CODECS[self.codec]

    def resolved_pix_fmt(self) -> str:
        if self.pix_fmt:
            return self.pix_fmt
        option = self.codec_option()
        if option.id == "prores":
            return "yuva444p10le" if self.prores_profile >= 4 else "yuv422p10le"
        return option.pix_fmts[0]


@dataclass
class ExportProgress:
    frame: int = 0
    total_frames: int = 0
    out_time: float = 0.0
    duration: float = 0.0
    fps: float = 0.0
    speed: str = ""
    done: bool = False

    @property
    def fraction(self) -> float:
        if self.total_frames:
            return min(1.0, self.frame / self.total_frames)
        if self.duration:
            return min(1.0, self.out_time / self.duration)
        return 0.0


@dataclass
class ExportResult:
    ok: bool
    output_path: str = ""
    cancelled: bool = False
    message: str = ""
    command: List[str] = field(default_factory=list)


ProgressCallback = Callable[[ExportProgress], None]

_PROGRESS_KEYS = re.compile(r"^(frame|fps|out_time_us|out_time_ms|speed|progress)=(.*)$")


class ExportJob:
    """Builds and runs one export.

    No Qt here - the UI runs this on a worker thread and forwards the progress
    callback into a signal.
    """

    def __init__(self, info: MediaInfo, params: PipelineParams,
                 settings: ExportSettings, tools: Optional[FFmpegTools] = None):
        self.info = info
        self.params = params.copy()
        self.settings = settings
        self.tools = tools or detect_tools()
        self._cancel = threading.Event()
        self._proc: Optional[subprocess.Popen] = None

    # -- validation ------------------------------------------------------
    def validate(self) -> List[str]:
        """Problems that would make this export fail, in plain language."""
        problems = list(self.tools.missing_requirements())
        option = self.settings.codec_option()
        if self.tools.available and not self.tools.has_encoder(option.encoder):
            problems.append(f"this ffmpeg build has no '{option.encoder}' encoder")
        if self.settings.container not in option.containers:
            problems.append(f"{option.label} cannot be written to "
                            f".{self.settings.container}")
        if not self.settings.output_path:
            problems.append("no output file chosen")
        elif os.path.abspath(self.settings.output_path) == os.path.abspath(self.info.path):
            problems.append("the output file would overwrite the source")
        return problems

    # -- command ---------------------------------------------------------
    def _video_args(self) -> List[str]:
        s = self.settings
        option = s.codec_option()
        args = ["-c:v", option.encoder]

        if option.rate_control == "crf":
            if s.bitrate_kbps:
                args += ["-b:v", f"{int(s.bitrate_kbps)}k"]
            else:
                args += ["-crf", str(quality_to_crf(s.quality))]
            args += ["-preset", s.preset]
            if option.encoder == "libx265":
                # Otherwise x265 prints a banner into the progress stream.
                args += ["-x265-params", "log-level=error"]
        else:
            args += ["-profile:v", str(int(s.prores_profile)), "-vendor", "apl0"]

        args += ["-pix_fmt", s.resolved_pix_fmt()]
        return args

    def _audio_args(self) -> List[str]:
        mode = self.settings.audio_mode
        if mode == "none" or not self.info.has_audio:
            return ["-an"]
        if mode == "aac":
            return ["-c:a", "aac", "-b:a", "320k"]
        return ["-c:a", "copy"]

    def build_command(self, lut_path: str) -> List[str]:
        s = self.settings
        out_size = None
        if s.width and s.height:
            out_size = (int(s.width), int(s.height))

        chain = export_chain(self.params.source, lut_path,
                             interp=s.lut_interp, out_size=out_size)

        args = [self.tools.ffmpeg or "ffmpeg", "-hide_banner", "-nostdin",
                "-loglevel", "error", "-nostats",
                "-progress", "pipe:1",
                "-y" if s.overwrite else "-n",
                "-i", self.info.path,
                # Video plus any audio; '?' makes audio optional so a silent
                # source is not an error. Exotic data streams are left out
                # deliberately - they are the usual cause of container errors.
                "-map", "0:v:0", "-map", "0:a?",
                "-filter:v", chain]
        args += self._video_args()
        args += output_tagging_args(self.params.output.profile_id)

        if s.fps:
            args += ["-r", f"{float(s.fps):.6f}"]
        else:
            # Copy the source's timing exactly rather than letting FFmpeg
            # resample it, which is what protects 23.976 and VFR footage.
            args += ["-fps_mode", "passthrough"]

        args += self._audio_args()
        if s.container == "mp4":
            args += ["-movflags", "+faststart"]
        args.append(s.output_path)
        return args

    # -- running ---------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self, on_progress: Optional[ProgressCallback] = None) -> ExportResult:
        """Bake the LUT, run FFmpeg, and report progress until it finishes."""
        problems = self.validate()
        if problems:
            return ExportResult(False, message="; ".join(problems))
        if not self.tools.available:
            raise FFmpegNotFound("ffmpeg is not available")

        lut_dir = tempfile.mkdtemp(prefix="vct-lut-")
        lut_path = os.path.join(lut_dir, "grade.cube")
        write_cube(lut_path, build_lut(self.params, self.settings.lut_size),
                   title=f"VCT {self.params.source.profile_id} to "
                         f"{self.params.output.profile_id}")

        command = self.build_command(lut_path)
        progress = ExportProgress(total_frames=self.info.estimated_frames,
                                  duration=self.info.duration)
        try:
            self._proc = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1)
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._cancel.is_set():
                    break
                if self._consume_progress(line, progress) and on_progress:
                    on_progress(progress)
            self._proc.wait()
            stderr = (self._proc.stderr.read() if self._proc.stderr else "") or ""
        finally:
            self._cleanup(lut_path, lut_dir)

        if self._cancel.is_set():
            self._remove_partial_output()
            return ExportResult(False, cancelled=True,
                                message="Export cancelled.", command=command)
        if self._proc.returncode != 0:
            tail = "\n".join(stderr.strip().splitlines()[-6:])
            raise FFmpegError(f"export failed (exit {self._proc.returncode})",
                              command, stderr or tail)

        progress.done = True
        if on_progress:
            on_progress(progress)
        return ExportResult(True, output_path=self.settings.output_path,
                            command=command)

    @staticmethod
    def _consume_progress(line: str, progress: ExportProgress) -> bool:
        match = _PROGRESS_KEYS.match(line.strip())
        if not match:
            return False
        key, value = match.group(1), match.group(2).strip()
        try:
            if key == "frame":
                progress.frame = int(value)
            elif key == "fps":
                progress.fps = float(value)
            elif key == "out_time_us":
                progress.out_time = int(value) / 1e6
            elif key == "out_time_ms":
                progress.out_time = int(value) / 1e6   # FFmpeg misnames this one
            elif key == "speed":
                progress.speed = value
            elif key == "progress":
                progress.done = value == "end"
        except ValueError:
            return False
        return True

    def _remove_partial_output(self) -> None:
        path = self.settings.output_path
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _cleanup(lut_path: str, lut_dir: str) -> None:
        for action, target in ((os.remove, lut_path), (os.rmdir, lut_dir)):
            try:
                action(target)
            except OSError:
                pass


def default_output_path(source_path: str, container: str = "mp4",
                        suffix: str = "_graded") -> str:
    """A sensible destination next to the source, never overwriting it."""
    base, _ = os.path.splitext(source_path)
    ext = CONTAINER_EXTENSIONS.get(container, ".mp4")
    candidate = f"{base}{suffix}{ext}"
    index = 2
    while os.path.exists(candidate):
        candidate = f"{base}{suffix}_{index}{ext}"
        index += 1
    return candidate
