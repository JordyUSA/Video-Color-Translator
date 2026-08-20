"""Small reusable controls."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QMouseEvent
from PySide6.QtWidgets import (QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QSlider, QToolButton, QVBoxLayout, QWidget)

from . import theme


class LabeledSlider(QWidget):
    """A float slider with a readout that can be typed into.

    Qt sliders are integer-only, so the value is kept in float and scaled on the
    way in and out.  Double-clicking the label resets to the default, which is
    the fastest way to undo a tweak you have talked yourself out of.
    """

    valueChanged = Signal(float)
    #: Emitted with True when the user starts dragging and False when they stop,
    #: so the preview can render a cheap draft frame in between.
    interactionChanged = Signal(bool)

    def __init__(self, label: str, minimum: float, maximum: float,
                 default: float = 0.0, decimals: int = 2, suffix: str = "",
                 tooltip: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._min, self._max = float(minimum), float(maximum)
        self._default = float(default)
        self._decimals = decimals
        self._steps = 1000
        self._suffix = suffix
        self._updating = False

        self.label = QLabel(label)
        self.label.setToolTip(tooltip or label)
        self.label.installEventFilter(self)
        self.label.setCursor(Qt.PointingHandCursor)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(self._min, self._max)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(max((self._max - self._min) / 100.0, 10 ** -decimals))
        self.spin.setValue(self._default)
        self.spin.setSuffix(suffix)
        self.spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.spin.setFixedWidth(62)
        self.spin.setAlignment(Qt.AlignRight)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setValue(self._to_slider(self._default))
        self.slider.setToolTip(tooltip or label)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.label, 1)
        top.addWidget(self.spin, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        layout.addLayout(top)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.sliderPressed.connect(lambda: self.interactionChanged.emit(True))
        self.slider.sliderReleased.connect(lambda: self.interactionChanged.emit(False))
        self.spin.valueChanged.connect(self._spin_changed)

    # -- conversion ------------------------------------------------------
    def _to_slider(self, value: float) -> int:
        span = self._max - self._min or 1.0
        return int(round((value - self._min) / span * self._steps))

    def _from_slider(self, position: int) -> float:
        span = self._max - self._min
        return self._min + (position / self._steps) * span

    # -- events ----------------------------------------------------------
    def eventFilter(self, obj, event) -> bool:
        if obj is self.label and isinstance(event, QMouseEvent) \
                and event.type() == QMouseEvent.Type.MouseButtonDblClick:
            self.reset()
            return True
        return super().eventFilter(obj, event)

    def _slider_moved(self, position: int) -> None:
        if self._updating:
            return
        self._updating = True
        value = self._from_slider(position)
        self.spin.setValue(value)
        self._updating = False
        self.valueChanged.emit(value)

    def _spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(self._to_slider(value))
        self._updating = False
        self.valueChanged.emit(value)

    # -- api -------------------------------------------------------------
    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, value: float, notify: bool = True) -> None:
        value = max(self._min, min(self._max, float(value)))
        if not notify:
            self._updating = True
        self.spin.setValue(value)
        self.slider.setValue(self._to_slider(value))
        if not notify:
            self._updating = False

    def reset(self) -> None:
        self.setValue(self._default)

    def isDefault(self) -> bool:
        return abs(self.value() - self._default) < 10 ** -(self._decimals + 1)


class Section(QFrame):
    """A titled, collapsible group of controls."""

    def __init__(self, title: str, parent: Optional[QWidget] = None,
                 collapsed: bool = False):
        super().__init__(parent)
        self.setObjectName("Section")
        self.setFrameShape(QFrame.NoFrame)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self._toggle.setStyleSheet(
            f"QToolButton {{ color: {theme.TEXT_DIM}; font-weight: 600; "
            f"border: none; padding: 6px 2px; }}")
        self._toggle.toggled.connect(self._on_toggled)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(2, 0, 2, 6)
        self.body_layout.setSpacing(4)
        self.body.setVisible(not collapsed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._toggle)
        outer.addWidget(self.body)

    def _on_toggled(self, checked: bool) -> None:
        self.body.setVisible(checked)
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def addWidget(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def addLayout(self, layout) -> None:
        self.body_layout.addLayout(layout)


class InfoTable(QWidget):
    """Two-column read-only label/value grid, for the source info panel."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(3)
        self._grid.setColumnStretch(1, 1)
        self._rows = 0

    def setRows(self, rows) -> None:
        self.clear()
        mono = QFont("monospace")
        mono.setStyleHint(QFont.Monospace)
        for label, value in rows:
            name = QLabel(str(label))
            name.setObjectName("Dim")
            content = QLabel(str(value))
            content.setFont(mono)
            content.setTextInteractionFlags(Qt.TextSelectableByMouse)
            content.setWordWrap(True)
            self._grid.addWidget(name, self._rows, 0, Qt.AlignTop)
            self._grid.addWidget(content, self._rows, 1)
            self._rows += 1

    def clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows = 0
