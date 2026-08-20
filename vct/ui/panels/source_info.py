"""Read-only panel showing what the file says about itself."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...media.detect import Detection
from ...media.probe import MediaInfo
from .. import theme
from ..widgets import InfoTable, Section


class SourceInfoPanel(QWidget):
    """File and colour metadata, plus how much the detection can be trusted."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.section = Section("SOURCE")

        self.table = InfoTable()
        self.detection_label = QLabel("No file open")
        self.detection_label.setWordWrap(True)
        self.detection_label.setObjectName("Dim")

        self.section.addWidget(self.table)
        self.section.addWidget(self.detection_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.section)

    def clear(self) -> None:
        self.table.clear()
        self.detection_label.setText("No file open")
        self.detection_label.setObjectName("Dim")

    def setSource(self, info: MediaInfo, detection: Detection) -> None:
        self.table.setRows(info.summary_rows())
        # A low-confidence guess is called out in warning colour rather than
        # blending in, because acting on it silently is the expensive mistake.
        highlight = not detection.is_trustworthy
        self.detection_label.setObjectName("Warning" if highlight else "Dim")
        prefix = "Interpreted as" if detection.is_trustworthy else "Best guess:"
        self.detection_label.setText(
            f"<b>{prefix} {detection.profile.label}</b><br>"
            f"<span style='color:{theme.TEXT_DIM}'>{detection.confidence_label}. "
            f"{detection.reason}</span>")
        self.detection_label.style().unpolish(self.detection_label)
        self.detection_label.style().polish(self.detection_label)
