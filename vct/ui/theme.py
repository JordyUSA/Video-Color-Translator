"""Dark theme.

Neutral greys throughout, deliberately: any tint in the surrounding chrome
biases how you judge the colour of the picture next to it. The one accent
colour is reserved for controls that are actively doing something.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette

BACKGROUND = "#1b1b1b"
PANEL = "#242424"
PANEL_RAISED = "#2c2c2c"
BORDER = "#383838"
TEXT = "#dcdcdc"
TEXT_DIM = "#8f8f8f"
ACCENT = "#4c9aff"
ACCENT_DIM = "#2f5f9e"
WARNING = "#e0a33e"
#: The preview surround. Mid grey rather than black so the eye has a stable
#: reference and highlights are not exaggerated by a black surround.
VIEWER_BACKGROUND = "#141414"

STYLESHEET = f"""
QWidget {{
    background: {BACKGROUND};
    color: {TEXT};
    font-size: 12px;
}}
QMainWindow::separator {{ background: {BORDER}; width: 1px; height: 1px; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BACKGROUND}; border: none; }}
QScrollBar:vertical {{ background: {BACKGROUND}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #4a4a4a; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 8px;
    padding: 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {TEXT_DIM};
}}

QLabel#SectionHeader {{ color: {TEXT_DIM}; font-weight: 600; letter-spacing: 0.5px; }}
QLabel#Dim {{ color: {TEXT_DIM}; }}
QLabel#Warning {{ color: {WARNING}; }}
QLabel#MonoValue {{ font-family: monospace; color: {TEXT}; }}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {PANEL_RAISED};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {ACCENT_DIM};
}}
QComboBox:hover, QLineEdit:hover {{ border-color: #4a4a4a; }}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_RAISED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}
QComboBox:disabled, QLineEdit:disabled, QPushButton:disabled {{ color: #5a5a5a; }}

QPushButton {{
    background: {PANEL_RAISED};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 5px 12px;
}}
QPushButton:hover {{ background: #343434; }}
QPushButton:pressed {{ background: #1f1f1f; }}
QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QPushButton#Primary {{ background: {ACCENT_DIM}; border-color: {ACCENT}; font-weight: 600; }}
QPushButton#Primary:hover {{ background: {ACCENT}; color: #10233d; }}

QSlider::groove:horizontal {{
    background: #171717; height: 3px; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; height: 3px; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: #cfcfcf; width: 11px; height: 11px;
    margin: -5px 0; border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::handle:horizontal:disabled {{ background: #4a4a4a; }}

QToolBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; spacing: 4px; padding: 4px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 3px; padding: 4px 8px; }}
QToolButton:hover {{ background: {PANEL_RAISED}; border-color: {BORDER}; }}
QToolButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {TEXT_DIM}; }}
QStatusBar::item {{ border: none; }}

QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item:selected {{ background: {PANEL_RAISED}; }}
QMenu {{ background: {PANEL_RAISED}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}

QProgressBar {{
    background: {PANEL_RAISED}; border: 1px solid {BORDER};
    border-radius: 3px; text-align: center; height: 18px;
}}
QProgressBar::chunk {{ background: {ACCENT_DIM}; border-radius: 2px; }}

QCheckBox::indicator {{
    width: 13px; height: 13px; border: 1px solid {BORDER};
    border-radius: 3px; background: {PANEL_RAISED};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QToolTip {{
    background: #101010; color: {TEXT};
    border: 1px solid {BORDER}; padding: 4px;
}}
"""


def apply_theme(app) -> None:
    """Apply the palette and stylesheet to a QApplication."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(PANEL_RAISED))
    palette.setColor(QPalette.AlternateBase, QColor(PANEL))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(PANEL_RAISED))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_DIM))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipBase, QColor("#101010"))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#5a5a5a"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5a5a5a"))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
