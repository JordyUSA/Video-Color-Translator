"""Export dialog and the thread that runs the render."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..core.pipeline import PipelineParams
from ..media.exporter import (CODECS, CONTAINER_EXTENSIONS, PRORES_PROFILES,
                              ExportJob, ExportProgress, ExportResult,
                              ExportSettings, default_output_path,
                              quality_to_crf)
from ..media.ffmpeg import FFmpegTools
from ..media.probe import MediaInfo
from .widgets import LabeledSlider

_X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"]


class ExportWorker(QThread):
    """Runs one :class:`ExportJob` off the UI thread."""

    progressed = Signal(object)
    finished_with = Signal(object)

    def __init__(self, job: ExportJob, parent=None):
        super().__init__(parent)
        self.job = job

    def run(self) -> None:                                # pragma: no cover - thread
        try:
            result = self.job.run(self.progressed.emit)
        except Exception as exc:                          # noqa: BLE001
            result = ExportResult(False, message=str(exc))
        self.finished_with.emit(result)

    def cancel(self) -> None:
        self.job.cancel()


class ExportDialog(QDialog):
    """Collects encoding settings and runs the export.

    Every field defaults to matching the source, so the shortest path through
    this dialog changes nothing but the colour.
    """

    def __init__(self, info: MediaInfo, params: PipelineParams,
                 tools: FFmpegTools, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(560)
        self.info = info
        self.params = params
        self.tools = tools
        self._worker: Optional[ExportWorker] = None

        # -- destination
        self.path_edit = QLineEdit(default_output_path(info.path, "mp4"))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        path_widget = QWidget()
        path_widget.setLayout(path_row)

        # -- codec
        self.codec_combo = QComboBox()
        for codec in CODECS.values():
            enabled = tools.has_encoder(codec.encoder)
            label = codec.label if enabled else f"{codec.label} (not in this build)"
            self.codec_combo.addItem(label, codec.id)
            if not enabled:
                self.codec_combo.model().item(self.codec_combo.count() - 1).setEnabled(False)

        self.container_combo = QComboBox()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(_X264_PRESETS)
        self.preset_combo.setCurrentText("medium")
        self.prores_combo = QComboBox()
        for value, label in PRORES_PROFILES.items():
            self.prores_combo.addItem(label, value)
        self.prores_combo.setCurrentIndex(3)

        self.quality_slider = LabeledSlider(
            "Quality", 0.0, 100.0, 70.0, 0, "",
            "Maps to CRF. Higher is better quality and a larger file.")
        self.crf_label = QLabel()
        self.crf_label.setObjectName("Dim")

        # -- geometry and timing, all defaulting to match source
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 16384)
        self.width_spin.setSpecialValueText("Match source")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 16384)
        self.height_spin.setSpecialValueText("Match source")
        size_row = QHBoxLayout()
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.height_spin)
        size_widget = QWidget()
        size_widget.setLayout(size_row)

        self.fps_label = QLabel(
            f"{info.frame_rate:.3f} fps - source timing preserved exactly")
        self.fps_label.setObjectName("Dim")
        self.fps_label.setToolTip(
            "No frame rate conversion is applied. Timestamps are passed through, "
            "so 23.976 and variable-rate footage come out unchanged.")

        self.audio_combo = QComboBox()
        self.audio_combo.addItem("Copy without re-encoding", "copy")
        self.audio_combo.addItem("Re-encode to AAC 320k", "aac")
        self.audio_combo.addItem("No audio", "none")
        self.audio_combo.setEnabled(info.has_audio)
        if not info.has_audio:
            self.audio_combo.setCurrentIndex(2)

        self.lut_combo = QComboBox()
        self.lut_combo.addItem("33 (standard)", 33)
        self.lut_combo.addItem("65 (high precision)", 65)
        self.lut_combo.setToolTip(
            "Size of the colour lookup table handed to FFmpeg. 65 is more "
            "accurate on heavily saturated wide-gamut footage and costs a "
            "moment longer to build.")

        form = QFormLayout()
        form.setSpacing(6)
        form.addRow("Output file", path_widget)
        form.addRow("Codec", self.codec_combo)
        form.addRow("Container", self.container_combo)
        form.addRow("Preset", self.preset_combo)
        form.addRow("ProRes profile", self.prores_combo)
        form.addRow("", self.quality_slider)
        form.addRow("", self.crf_label)
        form.addRow("Resolution", size_widget)
        form.addRow("Frame rate", self.fps_label)
        form.addRow("Audio", self.audio_combo)
        form.addRow("LUT size", self.lut_combo)

        # -- progress and the command, for transparency
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setObjectName("Dim")

        self.command_view = QPlainTextEdit()
        self.command_view.setReadOnly(True)
        self.command_view.setMaximumHeight(96)
        self.command_view.setVisible(False)
        self.command_toggle = QPushButton("Show FFmpeg command")
        self.command_toggle.setCheckable(True)
        self.command_toggle.toggled.connect(self._toggle_command)

        self.buttons = QDialogButtonBox()
        self.export_button = self.buttons.addButton("Export",
                                                    QDialogButtonBox.AcceptRole)
        self.export_button.setObjectName("Primary")
        self.cancel_button = self.buttons.addButton(QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._start)
        self.buttons.rejected.connect(self._reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.command_toggle)
        layout.addWidget(self.command_view)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)

        self.codec_combo.currentIndexChanged.connect(self._codec_changed)
        self.container_combo.currentIndexChanged.connect(self._sync_extension)
        self.quality_slider.valueChanged.connect(lambda _: self._update_quality())
        self._codec_changed()

    # -- settings --------------------------------------------------------
    def settings(self) -> ExportSettings:
        return ExportSettings(
            output_path=self.path_edit.text().strip(),
            codec=self.codec_combo.currentData() or "h265",
            container=self.container_combo.currentData() or "mp4",
            quality=self.quality_slider.value(),
            preset=self.preset_combo.currentText(),
            prores_profile=self.prores_combo.currentData() or 3,
            width=self.width_spin.value() or None,
            height=self.height_spin.value() or None,
            audio_mode=self.audio_combo.currentData() or "copy",
            lut_size=self.lut_combo.currentData() or 33,
        )

    # -- reactions -------------------------------------------------------
    def _codec_changed(self) -> None:
        codec = CODECS.get(self.codec_combo.currentData() or "h265")
        if codec is None:
            return
        current = self.container_combo.currentData()
        self.container_combo.blockSignals(True)
        self.container_combo.clear()
        for container in codec.containers:
            self.container_combo.addItem(f".{container}", container)
        index = self.container_combo.findData(current)
        self.container_combo.setCurrentIndex(max(index, 0))
        self.container_combo.blockSignals(False)

        is_crf = codec.rate_control == "crf"
        self.quality_slider.setVisible(is_crf)
        self.crf_label.setVisible(is_crf)
        self.preset_combo.setVisible(is_crf)
        self.prores_combo.setVisible(not is_crf)
        self._sync_extension()
        self._update_quality()

    def _sync_extension(self) -> None:
        container = self.container_combo.currentData() or "mp4"
        wanted = CONTAINER_EXTENSIONS.get(container, ".mp4")
        base, ext = os.path.splitext(self.path_edit.text().strip())
        if base and ext.lower() != wanted:
            self.path_edit.setText(base + wanted)

    def _update_quality(self) -> None:
        self.crf_label.setText(f"CRF {quality_to_crf(self.quality_slider.value())} "
                               f"- lower is better quality")

    def _toggle_command(self, shown: bool) -> None:
        self.command_toggle.setText("Hide FFmpeg command" if shown
                                    else "Show FFmpeg command")
        if shown:
            job = ExportJob(self.info, self.params, self.settings(), self.tools)
            command = job.build_command("<generated>.cube")
            self.command_view.setPlainText(" ".join(
                f"'{part}'" if " " in part else part for part in command))
        self.command_view.setVisible(shown)

    def _browse(self) -> None:
        container = self.container_combo.currentData() or "mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to", self.path_edit.text(),
            f"Video (*{CONTAINER_EXTENSIONS.get(container, '.mp4')});;All files (*)")
        if path:
            self.path_edit.setText(path)

    # -- running ---------------------------------------------------------
    def _start(self) -> None:
        job = ExportJob(self.info, self.params, self.settings(), self.tools)
        problems = job.validate()
        if problems:
            QMessageBox.warning(self, "Cannot export",
                                "\n".join(f"- {p}" for p in problems))
            return
        if os.path.exists(job.settings.output_path):
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"{os.path.basename(job.settings.output_path)} already exists. "
                f"Replace it?")
            if reply != QMessageBox.Yes:
                return

        self.export_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.status.setText("Rendering...")

        self._worker = ExportWorker(job, self)
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished_with.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, progress: ExportProgress) -> None:
        self.progress.setValue(int(progress.fraction * 100))
        parts = [f"Frame {progress.frame}"]
        if progress.total_frames:
            parts[0] += f" of {progress.total_frames}"
        if progress.speed:
            parts.append(f"{progress.speed}")
        self.status.setText("  -  ".join(parts))

    def _on_finished(self, result: ExportResult) -> None:
        self.export_button.setEnabled(True)
        self._worker = None
        if result.ok:
            self.progress.setValue(100)
            self.status.setText(f"Done: {result.output_path}")
            QMessageBox.information(self, "Export complete",
                                    f"Written to\n{result.output_path}")
            self.accept()
        elif result.cancelled:
            self.status.setText("Cancelled.")
            self.progress.setVisible(False)
        else:
            self.progress.setVisible(False)
            self.status.setText("Failed.")
            QMessageBox.critical(self, "Export failed", result.message)

    def _reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(4000)
            self.status.setText("Cancelled.")
            return
        self.reject()
