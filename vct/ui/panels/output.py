"""Output panel: delivery space and how HDR gets there."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QFormLayout, QLabel, QVBoxLayout,
                               QWidget)

from ...core.camera_profiles import OUTPUT_PROFILES, get_output_profile
from ...core.tonemap import TONE_MAPPERS, get_tone_mapper
from ..widgets import LabeledSlider, Section


class OutputPanel(QWidget):
    """Target colour space, tone mapping operator and its two shared controls."""

    changed = Signal()
    interactionChanged = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.section = Section("OUTPUT")

        self.profile_combo = QComboBox()
        for profile in OUTPUT_PROFILES.values():
            self.profile_combo.addItem(profile.label, profile.id)
        self.profile_combo.setToolTip(
            "The space the preview is shown in and the export is tagged for.")

        self.mapper_combo = QComboBox()
        for mapper in TONE_MAPPERS.values():
            self.mapper_combo.addItem(mapper.label, mapper.id)
        self.mapper_combo.setToolTip(
            "How out-of-range highlights are brought into the display's range.")

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Luminance (keeps hue)", "luminance")
        self.mode_combo.addItem("Per channel (film-like)", "rgb")
        self.mode_combo.setToolTip(
            "Luminance scales all three channels together, so colours keep their "
            "hue. Per channel lets bright colours desaturate toward white, which "
            "is how film behaves and often looks more natural on skin.")

        self.desat_slider = LabeledSlider(
            "Highlight desaturation", 0.0, 1.0, 0.35, 2, "",
            "Rolls saturated highlights toward white before the tone curve, so "
            "a bright red light becomes a glow rather than a flat red patch.")

        self.gamut_slider = LabeledSlider(
            "Gamut compression", 0.0, 1.0, 1.0, 2, "",
            "Folds colours outside the target gamut back in rather than clipping "
            "them. Applies only when the source gamut is wider than the output.")

        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("Dim")

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(5)
        form.addRow("Colour space", self.profile_combo)
        form.addRow("Tone mapping", self.mapper_combo)
        form.addRow("Applied by", self.mode_combo)

        self.section.addLayout(form)
        self.section.addWidget(self.desat_slider)
        self.section.addWidget(self.gamut_slider)
        self.section.addWidget(self.note)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

        for combo in (self.profile_combo, self.mapper_combo, self.mode_combo):
            combo.currentIndexChanged.connect(self._emit)
        for slider in (self.desat_slider, self.gamut_slider):
            slider.valueChanged.connect(lambda _: self._emit())
            slider.interactionChanged.connect(self.interactionChanged)

        self._update_note()

    # -- state -----------------------------------------------------------
    def profileId(self) -> str:
        return self.profile_combo.currentData() or "rec709"

    def toneMapper(self) -> str:
        return self.mapper_combo.currentData() or "bt2390"

    def toneMode(self) -> str:
        return self.mode_combo.currentData() or "luminance"

    def desaturation(self) -> float:
        return self.desat_slider.value()

    def gamutCompression(self) -> Optional[float]:
        return self.gamut_slider.value()

    def setHdrSource(self, is_hdr: bool) -> None:
        """Tone mapping only does anything when there is range to bring down."""
        for widget in (self.mapper_combo, self.mode_combo, self.desat_slider):
            widget.setEnabled(is_hdr)
        self._update_note(is_hdr)

    def _update_note(self, is_hdr: Optional[bool] = None) -> None:
        parts = []
        try:
            parts.append(get_output_profile(self.profileId()).note)
            if is_hdr is not False:
                parts.append(get_tone_mapper(self.toneMapper()).note)
        except KeyError:
            pass
        self.note.setText(" ".join(p for p in parts if p))

    def _emit(self) -> None:
        self._update_note()
        self.changed.emit()
