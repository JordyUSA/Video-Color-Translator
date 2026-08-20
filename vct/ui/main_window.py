"""The main window: preview, transport, and the control column."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QScrollArea, QSlider,
                               QSizePolicy, QStatusBar, QVBoxLayout, QWidget)

from ..core.lut import DEFAULT_LUT_SIZE, build_lut, write_cube
from ..core.pipeline import (ColorPipeline, OutputParams, PipelineParams,
                             SourceParams)
from ..media.detect import detect, suggested_peak_nits
from ..media.ffmpeg import FFmpegError, FFmpegNotFound, detect_tools
from ..media.probe import MediaInfo, probe
from . import theme
from .export_dialog import ExportDialog
from .panels import AdjustPanel, InterpretPanel, OutputPanel, SourceInfoPanel
from .playback import PlaybackThread
from .preview import create_preview

#: How long after the last slider movement to re-render at full quality.
SETTLE_MS = 130

VIDEO_FILTER = ("Video files (*.mov *.mp4 *.mxf *.mkv *.m4v *.avi *.braw *.r3d "
                "*.mts *.m2ts *.webm);;All files (*)")


class MainWindow(QMainWindow):
    """Preview on the left, controls on the right, transport underneath."""

    def __init__(self, initial_file: Optional[str] = None,
                 force_cpu_preview: bool = False, lut_size: int = DEFAULT_LUT_SIZE):
        super().__init__()
        self.setWindowTitle("Video Color Translator")
        self.resize(1440, 860)

        self.tools = detect_tools()
        self.info: Optional[MediaInfo] = None
        self.playback: Optional[PlaybackThread] = None
        self._frame: Optional[np.ndarray] = None
        self._lut_size = lut_size
        self._bypass = False
        self._suppress_lut_rebuild = False

        self.preview, self.preview_backend = create_preview(
            force_cpu=force_cpu_preview)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if hasattr(self.preview, "initialisationFailed"):
            self.preview.initialisationFailed.connect(self._on_gl_failed)

        self._build_transport()
        self._build_panels()
        self._build_layout()
        self._build_menu()

        self.settle_timer = QTimer(self)
        self.settle_timer.setSingleShot(True)
        self.settle_timer.setInterval(SETTLE_MS)
        self.settle_timer.timeout.connect(self._settle)

        self.setStatusBar(QStatusBar())
        self._report_environment()
        self._rebuild_lut()
        self.setAcceptDrops(True)

        if initial_file:
            self.openFile(initial_file)

    # ------------------------------------------------------------------ UI
    def _build_transport(self) -> None:
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_play)

        self.prev_button = QPushButton("<")
        self.prev_button.setToolTip("Previous frame (Left)")
        self.prev_button.setFixedWidth(32)
        self.prev_button.clicked.connect(lambda: self._step(-1))
        self.next_button = QPushButton(">")
        self.next_button.setToolTip("Next frame (Right)")
        self.next_button.setFixedWidth(32)
        self.next_button.clicked.connect(lambda: self._step(1))

        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.setEnabled(False)
        self.scrub.sliderMoved.connect(self._scrubbed)
        self.scrub.sliderPressed.connect(lambda: self._set_draft(True))
        self.scrub.sliderReleased.connect(lambda: self._set_draft(False))

        self.timecode = QLabel("00:00:00:00")
        self.timecode.setObjectName("MonoValue")
        self.timecode.setFixedWidth(96)
        self.timecode.setAlignment(Qt.AlignCenter)

        self.bypass_button = QPushButton("Bypass")
        self.bypass_button.setCheckable(True)
        self.bypass_button.setToolTip(
            "Show the source untouched, for comparison (B). The most useful "
            "control here - a grade is judged against what it started from.")
        self.bypass_button.toggled.connect(self._set_bypass)

        row = QHBoxLayout()
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(6)
        for widget in (self.play_button, self.prev_button, self.next_button):
            row.addWidget(widget)
        row.addWidget(self.scrub, 1)
        row.addWidget(self.timecode)
        row.addWidget(self.bypass_button)
        self.transport = QWidget()
        self.transport.setLayout(row)
        self.transport.setStyleSheet(
            f"background: {theme.PANEL}; border-top: 1px solid {theme.BORDER};")

    def _build_panels(self) -> None:
        self.source_panel = SourceInfoPanel()
        self.interpret_panel = InterpretPanel()
        self.output_panel = OutputPanel()
        self.adjust_panel = AdjustPanel()

        for panel in (self.interpret_panel, self.output_panel, self.adjust_panel):
            panel.changed.connect(self._rebuild_lut)
            panel.interactionChanged.connect(self._set_draft)
        self.interpret_panel.decodeAffected.connect(self._reinterpret_source)

        self.save_lut_button = QPushButton("Save .cube LUT...")
        self.save_lut_button.setToolTip(
            "Write the current transform as a 3D LUT, usable in Resolve, "
            "Premiere or any other tool that loads .cube files.")
        self.save_lut_button.clicked.connect(self.saveLut)

        self.export_button = QPushButton("Export video...")
        self.export_button.setObjectName("Primary")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.openExportDialog)

        column = QVBoxLayout()
        column.setContentsMargins(10, 8, 10, 10)
        column.setSpacing(6)
        for panel in (self.source_panel, self.interpret_panel,
                      self.output_panel, self.adjust_panel):
            column.addWidget(panel)
        column.addStretch(1)
        column.addWidget(self.save_lut_button)
        column.addWidget(self.export_button)

        holder = QWidget()
        holder.setLayout(column)
        self.sidebar = QScrollArea()
        self.sidebar.setWidget(holder)
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setFixedWidth(340)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def _build_layout(self) -> None:
        viewer = QVBoxLayout()
        viewer.setContentsMargins(0, 0, 0, 0)
        viewer.setSpacing(0)
        viewer.addWidget(self.preview, 1)
        viewer.addWidget(self.transport)
        viewer_widget = QWidget()
        viewer_widget.setLayout(viewer)

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(viewer_widget, 1)
        root.addWidget(self.sidebar)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open video...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.promptOpen)
        file_menu.addAction(open_action)

        self.export_action = QAction("&Export video...", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.openExportDialog)
        file_menu.addAction(self.export_action)

        save_lut = QAction("Save &LUT...", self)
        save_lut.setShortcut("Ctrl+L")
        save_lut.triggered.connect(self.saveLut)
        file_menu.addAction(save_lut)
        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        bypass = QAction("&Bypass grade", self)
        bypass.setShortcut("B")
        bypass.setCheckable(True)
        bypass.toggled.connect(self.bypass_button.setChecked)
        self.bypass_button.toggled.connect(bypass.setChecked)
        view_menu.addAction(bypass)

        for label, shortcut, delta in (("Next frame", "Right", 1),
                                       ("Previous frame", "Left", -1)):
            action = QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _=False, d=delta: self._step(d))
            view_menu.addAction(action)

        play = QAction("Play / pause", self)
        play.setShortcut("Space")
        play.triggered.connect(self._toggle_play)
        view_menu.addAction(play)

    def _report_environment(self) -> None:
        problems = self.tools.missing_requirements()
        backend = ("GPU (OpenGL)" if self.preview_backend == "opengl"
                   else "CPU (no OpenGL context available)")
        if problems:
            self.statusBar().showMessage(
                "FFmpeg problem: " + "; ".join(problems))
        else:
            version = self.tools.version.split(" Copyright")[0]
            self.statusBar().showMessage(f"{version}  -  preview: {backend}")

    # -------------------------------------------------------------- params
    def pipelineParams(self) -> PipelineParams:
        """Assemble the current state of every panel into one parameter block."""
        source = SourceParams(
            profile_id=self.interpret_panel.profileId(),
            transfer=self.interpret_panel.transferOverride(),
            primaries=self.interpret_panel.primariesOverride(),
            source_range=self.interpret_panel.sourceRange(),
            bit_depth=self.info.bit_depth if self.info else 10,
            source_peak_nits=self.interpret_panel.peakNits(),
        )
        output = OutputParams(
            profile_id=self.output_panel.profileId(),
            tone_mapper=self.output_panel.toneMapper(),
            tone_mode=self.output_panel.toneMode(),
            highlight_desaturation=self.output_panel.desaturation(),
            gamut_compression=self.output_panel.gamutCompression(),
        )
        params = PipelineParams(source=source, output=output,
                                grade=self.adjust_panel.gradeParams())
        params.bypass = self._bypass
        return params

    def _rebuild_lut(self) -> None:
        """Rebake the LUT and hand it to the preview.

        This is the whole interaction loop: a few milliseconds of NumPy over
        36 000 samples, independent of the video's resolution.
        """
        if self._suppress_lut_rebuild:
            return
        params = self.pipelineParams()
        self.output_panel.setHdrSource(ColorPipeline(params).is_hdr_source)
        try:
            lut = build_lut(params, self._lut_size)
        except (KeyError, ValueError) as exc:
            self.statusBar().showMessage(f"Cannot build LUT: {exc}")
            return
        self.preview.setLut(lut)

    def _set_draft(self, active: bool) -> None:
        self.preview.setDraft(active)
        if active:
            self.settle_timer.stop()
        else:
            self.settle_timer.start()

    def _settle(self) -> None:
        self.preview.setDraft(False)

    def _set_bypass(self, enabled: bool) -> None:
        self._bypass = enabled
        self._rebuild_lut()

    def _on_gl_failed(self, message: str) -> None:
        """Swap to the CPU preview if the GL path cannot run on this machine."""
        self.statusBar().showMessage(
            f"OpenGL preview unavailable ({message}); using the CPU preview.")
        old = self.preview
        self.preview, self.preview_backend = create_preview(force_cpu=True)
        layout = old.parentWidget().layout()
        layout.replaceWidget(old, self.preview)
        old.deleteLater()
        self._rebuild_lut()
        if self._frame is not None:
            self.preview.setFrame(self._frame)

    # ---------------------------------------------------------------- file
    def promptOpen(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if path:
            self.openFile(path)

    def openFile(self, path: str) -> None:
        if not self.tools.available:
            QMessageBox.critical(self, "FFmpeg not found",
                                 "FFmpeg could not be located. Install it with "
                                 "your package manager, or set VCT_FFMPEG to a "
                                 "binary.")
            return
        try:
            info = probe(path)
        except FileNotFoundError:
            QMessageBox.warning(self, "Not found", f"{path} does not exist.")
            return
        except (FFmpegError, FFmpegNotFound) as exc:
            QMessageBox.critical(self, "Cannot open file", str(exc))
            return

        self._stop_playback()
        self.info = info
        detection = detect(info)

        # Fill the panels in before rebuilding, so one LUT bake covers the lot.
        self._suppress_lut_rebuild = True
        self.interpret_panel.setDetection(
            detection, suggested_peak_nits(info), info.range_id)
        self.source_panel.setSource(info, detection)
        self._suppress_lut_rebuild = False
        self._rebuild_lut()

        self.setWindowTitle(f"{os.path.basename(path)} - Video Color Translator")
        for widget in (self.play_button, self.scrub, self.export_button):
            widget.setEnabled(True)
        self.export_action.setEnabled(True)

        self._start_playback()
        if not detection.is_trustworthy:
            self.statusBar().showMessage(
                f"Colour interpretation is a guess: {detection.profile.label}. "
                f"Check it against the picture.")

    def _start_playback(self) -> None:
        params = self.pipelineParams()
        self.playback = PlaybackThread(self.info, params.source, self.tools)
        self.playback.frameReady.connect(self._on_frame)
        self.playback.playbackStopped.connect(
            lambda: self.play_button.setText("Play"))
        self.playback.error.connect(
            lambda msg: self.statusBar().showMessage(f"Decode error: {msg}"))
        self.playback.start()
        self.playback.seek(0.0)

    def _stop_playback(self) -> None:
        if self.playback is not None:
            self.playback.stop()
            self.playback = None
        self._frame = None
        self.preview.setFrame(None)

    def _reinterpret_source(self) -> None:
        """Matrix or range changed, so the frame has to be decoded again."""
        if self.playback is not None:
            self.playback.setSourceParams(self.pipelineParams().source)

    # ------------------------------------------------------------ playback
    def _on_frame(self, frame: np.ndarray, position: float) -> None:
        self._frame = frame
        self.preview.setFrame(frame)
        self._update_timecode(position)
        if self.info and self.info.duration and not self.scrub.isSliderDown():
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(position / self.info.duration * 1000))
            self.scrub.blockSignals(False)

    def _update_timecode(self, position: float) -> None:
        rate = (self.info.frame_rate if self.info else 25.0) or 25.0
        hours, rem = divmod(int(position), 3600)
        minutes, seconds = divmod(rem, 60)
        frames = int((position - int(position)) * rate)
        self.timecode.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}")

    def _toggle_play(self) -> None:
        if self.playback is None:
            return
        self.playback.toggle()
        self.play_button.setText("Pause" if self.playback.is_playing else "Play")

    def _step(self, frames: int) -> None:
        if self.playback is not None:
            self.playback.step(frames)
            self.play_button.setText("Play")

    def _scrubbed(self, value: int) -> None:
        if self.playback is not None and self.info and self.info.duration:
            self.playback.seek(value / 1000.0 * self.info.duration)

    # -------------------------------------------------------------- output
    def saveLut(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save LUT", "grade.cube", "Cube LUT (*.cube)")
        if not path:
            return
        params = self.pipelineParams()
        params.bypass = False        # saving a bypassed grade would be useless
        try:
            write_cube(path, build_lut(params, self._lut_size),
                       title=f"{params.source.profile_id} to "
                             f"{params.output.profile_id}")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save LUT", str(exc))
            return
        self.statusBar().showMessage(f"LUT written to {path}")

    def openExportDialog(self) -> None:
        if self.info is None:
            return
        was_playing = self.playback is not None and self.playback.is_playing
        if was_playing:
            self.playback.pause()
        params = self.pipelineParams()
        params.bypass = False        # never export the bypass view
        ExportDialog(self.info, params, self.tools, self).exec()

    # -------------------------------------------------------------- events
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.openFile(urls[0].toLocalFile())

    def closeEvent(self, event) -> None:
        self._stop_playback()
        super().closeEvent(event)
