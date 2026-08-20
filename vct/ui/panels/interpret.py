"""Interpretation panel: how the incoming file should be read.

The equivalent of Premiere's "Interpret Footage". Nothing here changes a pixel
in the file; it changes what the pixels are taken to mean.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from ...core.camera_profiles import GROUP_ORDER, PROFILES, get_profile
from ...core.colorimetry import COLOR_SPACES
from ...core.transfer import TRANSFERS
from ...media.detect import Detection
from .. import theme
from ..widgets import LabeledSlider, Section


class InterpretPanel(QWidget):
    """Source profile, per-component overrides, range, and HDR source peak."""

    changed = Signal()
    interactionChanged = Signal(bool)
    #: Emitted when a change needs the frame decoded again, rather than just
    #: relit by a new LUT - only the matrix and range affect the decode.
    decodeAffected = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._loading = False
        self._detection: Optional[Detection] = None

        self.section = Section("INTERPRET AS")

        self.profile_combo = QComboBox()
        self._populate_profiles()
        self.profile_combo.setToolTip(
            "How to read the incoming file. Camera LOG entries set both the "
            "curve and the gamut, because one without the other is wrong.")

        self.reset_button = QPushButton("Use detected")
        self.reset_button.setToolTip("Go back to what the file's metadata says.")

        self.advanced_check = QCheckBox("Override transfer and gamut separately")
        self.advanced_check.setToolTip(
            "For footage that does not match any preset - a LOG curve recorded "
            "in an unusual gamut, for instance.")

        self.transfer_combo = QComboBox()
        for kind in ("sdr", "hdr", "log", "linear"):
            for tf in TRANSFERS.values():
                if tf.kind == kind:
                    self.transfer_combo.addItem(tf.label, tf.id)
        self.primaries_combo = QComboBox()
        for space in COLOR_SPACES.values():
            self.primaries_combo.addItem(space.label, space.id)

        self.range_combo = QComboBox()
        self.range_combo.addItem("Limited / video (16-235)", "tv")
        self.range_combo.addItem("Full / data (0-255)", "pc")
        self.range_combo.setToolTip(
            "Which range the file's code values occupy. Getting this wrong on "
            "LOG footage costs about a stop of shadow detail.")

        self.peak_slider = LabeledSlider(
            "Source peak", 100.0, 10000.0, 1000.0, 0, " nits",
            "Peak brightness the HDR master was graded for. Filled in from the "
            "file's mastering metadata when it has any.")

        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setObjectName("Dim")

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(5)
        form.addRow("Profile", self.profile_combo)
        form.addRow("Transfer", self.transfer_combo)
        form.addRow("Gamut", self.primaries_combo)
        form.addRow("Range", self.range_combo)

        self.section.addLayout(form)
        self.section.addWidget(self.advanced_check)
        self.section.addWidget(self.peak_slider)
        self.section.addWidget(self.notes)
        self.section.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        self.profile_combo.currentIndexChanged.connect(self._profile_changed)
        self.advanced_check.toggled.connect(self._advanced_toggled)
        self.transfer_combo.currentIndexChanged.connect(self._emit)
        self.primaries_combo.currentIndexChanged.connect(self._emit)
        self.range_combo.currentIndexChanged.connect(self._range_changed)
        self.peak_slider.valueChanged.connect(lambda _: self._emit())
        self.peak_slider.interactionChanged.connect(self.interactionChanged)
        self.reset_button.clicked.connect(self.useDetected)

        self._advanced_toggled(False)
        self._update_notes()

    # -- population ------------------------------------------------------
    def _populate_profiles(self) -> None:
        for group in GROUP_ORDER:
            members = [p for p in PROFILES.values() if p.group == group]
            if not members:
                continue
            self.profile_combo.insertSeparator(self.profile_combo.count())
            index = self.profile_combo.count()
            self.profile_combo.addItem(f"--- {group} ---", None)
            self.profile_combo.model().item(index).setEnabled(False)
            for profile in members:
                self.profile_combo.addItem(profile.label, profile.id)

    def _select_profile(self, profile_id: str) -> None:
        index = self.profile_combo.findData(profile_id)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)

    # -- state -----------------------------------------------------------
    def profileId(self) -> str:
        data = self.profile_combo.currentData()
        return data or "rec709"

    def transferOverride(self) -> Optional[str]:
        if not self.advanced_check.isChecked():
            return None
        return self.transfer_combo.currentData()

    def primariesOverride(self) -> Optional[str]:
        if not self.advanced_check.isChecked():
            return None
        return self.primaries_combo.currentData()

    def sourceRange(self) -> str:
        return self.range_combo.currentData() or "tv"

    def peakNits(self) -> float:
        return self.peak_slider.value()

    def setDetection(self, detection: Detection, peak_nits: float,
                     source_range: str) -> None:
        self._detection = detection
        self._loading = True
        self._select_profile(detection.profile_id)
        self.peak_slider.setValue(peak_nits, notify=False)
        index = self.range_combo.findData(source_range)
        if index >= 0:
            self.range_combo.setCurrentIndex(index)
        self.advanced_check.setChecked(False)
        self._sync_advanced_to_profile()
        self._loading = False
        self._update_notes()

    def useDetected(self) -> None:
        if self._detection is not None:
            self._select_profile(self._detection.profile_id)
            self.advanced_check.setChecked(False)

    # -- reactions -------------------------------------------------------
    def _sync_advanced_to_profile(self) -> None:
        """Keep the override combos showing the profile's own values, so
        switching to manual starts from where you already are."""
        try:
            profile = get_profile(self.profileId())
        except KeyError:
            return
        for combo, value in ((self.transfer_combo, profile.transfer),
                             (self.primaries_combo, profile.primaries)):
            index = combo.findData(value)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

    def _profile_changed(self) -> None:
        if self.profile_combo.currentData() is None:
            return
        self._sync_advanced_to_profile()
        self._update_notes()
        # Changing gamut changes the YUV matrix, which is applied by FFmpeg.
        self._emit(decode=True)

    def _advanced_toggled(self, checked: bool) -> None:
        self.transfer_combo.setEnabled(checked)
        self.primaries_combo.setEnabled(checked)
        if not self._loading:
            self._update_notes()
            self._emit(decode=True)

    def _range_changed(self) -> None:
        self._emit(decode=True)

    def _update_notes(self) -> None:
        parts = []
        try:
            profile = get_profile(self.profileId())
            if profile.note:
                parts.append(profile.note)
            transfer = TRANSFERS.get(self.transferOverride() or profile.transfer)
            if transfer is not None and transfer.note:
                parts.append(transfer.note)
            is_hdr = transfer is not None and transfer.kind == "hdr"
        except KeyError:
            is_hdr = False
        self.peak_slider.setVisible(is_hdr)
        self.notes.setText(" ".join(parts))

    def _emit(self, decode: bool = False) -> None:
        if self._loading:
            return
        self._update_notes()
        if decode:
            self.decodeAffected.emit()
        self.changed.emit()
