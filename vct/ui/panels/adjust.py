"""Adjustment panel: the primary grade controls."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ...core.grade import GradeParams
from ..widgets import LabeledSlider, Section

#: name, label, min, max, default, decimals, suffix, tooltip
_CONTROLS = [
    ("exposure", "Exposure", -5.0, 5.0, 0.0, 2, " st",
     "Linear gain, in stops. Applied before tone mapping, so pushing into the "
     "highlights rolls off rather than clipping."),
    ("contrast", "Contrast", -1.0, 1.0, 0.0, 2, "",
     "Slope around 18% grey in log space. Mid grey does not move."),
    ("saturation", "Saturation", -1.0, 1.0, 0.0, 2, "",
     "Distance from neutral in the output signal. -1 is monochrome."),
    ("temperature", "Temperature", -100.0, 100.0, 0.0, 0, "",
     "Warms or cools by adapting from a different assumed light source. "
     "Positive is warmer."),
    ("tint", "Tint", -100.0, 100.0, 0.0, 0, "",
     "Green to magenta, perpendicular to the temperature axis. Positive is "
     "magenta."),
    ("highlights", "Highlights", -2.0, 2.0, 0.0, 2, " st",
     "Gain on the brighter half of the image, masked by luminance."),
    ("shadows", "Shadows", -2.0, 2.0, 0.0, 2, " st",
     "Gain on the darker half. A gain rather than a lift, so black stays black."),
    ("black_point", "Black point", -0.2, 0.2, 0.0, 3, "",
     "Where the output signal's black lands. Applied last."),
    ("white_point", "White point", 0.5, 1.5, 1.0, 3, "",
     "Where the output signal's white lands. Applied last."),
]

_ORDER_NOTE = (
    "Applied in order: exposure, white balance, contrast and highlight/shadow "
    "recovery on linear light before tone mapping; saturation and the black/"
    "white trim on the output signal after it."
)


class AdjustPanel(QWidget):
    """The nine primary controls, plus a reset."""

    changed = Signal()
    interactionChanged = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.section = Section("ADJUSTMENTS")
        self._sliders = {}

        for name, label, low, high, default, decimals, suffix, tip in _CONTROLS:
            slider = LabeledSlider(label, low, high, default, decimals, suffix, tip)
            slider.valueChanged.connect(lambda _v: self.changed.emit())
            slider.interactionChanged.connect(self.interactionChanged)
            self._sliders[name] = slider
            self.section.addWidget(slider)

        note = QLabel(_ORDER_NOTE)
        note.setWordWrap(True)
        note.setObjectName("Dim")
        self.section.addWidget(note)

        self.reset_button = QPushButton("Reset adjustments")
        self.reset_button.clicked.connect(self.reset)
        self.section.addWidget(self.reset_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

    def gradeParams(self) -> GradeParams:
        return GradeParams(**{name: slider.value()
                              for name, slider in self._sliders.items()})

    def setGradeParams(self, grade: GradeParams) -> None:
        for name, slider in self._sliders.items():
            slider.setValue(getattr(grade, name), notify=False)
        self.changed.emit()

    def reset(self) -> None:
        for slider in self._sliders.values():
            slider.setValue(slider._default, notify=False)
        self.changed.emit()

    def isNeutral(self) -> bool:
        return all(slider.isDefault() for slider in self._sliders.values())
