"""UI smoke tests, run offscreen.

These do not check that anything looks right - no display here - but they do
check that the window builds, that every control is wired to the pipeline, and
that the interaction loop stays fast enough to drag a slider through.
"""

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication          # noqa: E402

from vct.core.grade import GradeParams              # noqa: E402
from vct.core.lut import build_lut, identity_lut    # noqa: E402
from vct.media.detect import detect                 # noqa: E402
from vct.media.probe import MediaInfo, probe        # noqa: E402

from .conftest import requires_ffmpeg               # noqa: E402


@pytest.fixture(scope="module")
def app():
    """One offscreen QApplication for the module."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if existing is not None:
        return existing
    from vct.ui.theme import apply_theme
    application = QApplication([])
    apply_theme(application)
    return application


@pytest.fixture
def window(app):
    from vct.ui.main_window import MainWindow
    win = MainWindow(force_cpu_preview=True)
    yield win
    win.close()


# ------------------------------------------------------------------ widgets

def test_labeled_slider_round_trips(app):
    from vct.ui.widgets import LabeledSlider
    slider = LabeledSlider("Exposure", -5.0, 5.0, 0.0, 2, " st")
    assert slider.isDefault()
    slider.setValue(2.25)
    assert slider.value() == pytest.approx(2.25, abs=0.02)
    assert not slider.isDefault()
    slider.reset()
    assert slider.isDefault() and slider.value() == 0.0


def test_labeled_slider_clamps_out_of_range(app):
    from vct.ui.widgets import LabeledSlider
    slider = LabeledSlider("Contrast", -1.0, 1.0, 0.0, 2)
    slider.setValue(99.0)
    assert slider.value() == pytest.approx(1.0)


def test_slider_emits_interaction_for_the_draft_path(app):
    from vct.ui.widgets import LabeledSlider
    slider = LabeledSlider("Exposure", -5.0, 5.0)
    seen = []
    slider.interactionChanged.connect(seen.append)
    slider.slider.sliderPressed.emit()
    slider.slider.sliderReleased.emit()
    assert seen == [True, False]


# ------------------------------------------------------------------ preview

def test_cpu_preview_renders_a_frame(app):
    from vct.ui.preview import CpuPreview
    preview = CpuPreview()
    assert not preview.hasContent
    frame = (np.random.default_rng(0).random((90, 160, 3)) * 65535).astype(np.uint16)
    preview.setFrame(frame)
    preview.setLut(identity_lut(17))
    assert preview.hasContent
    image = preview._render()
    assert image is not None and (image.width(), image.height()) == (160, 90)


def test_preview_keeps_aspect_ratio(app):
    from vct.ui.preview import CpuPreview
    preview = CpuPreview()
    rect = preview._target_rect(1000, 1000, 1920, 1080)
    assert rect.width() == pytest.approx(1000.0)
    assert rect.height() == pytest.approx(1000 * 1080 / 1920)
    assert rect.y() == pytest.approx((1000 - rect.height()) / 2)


def test_identity_lut_preview_returns_the_source(app):
    """A neutral grade must not alter the picture on the way to the screen."""
    from vct.ui.preview import CpuPreview
    from vct.core.lut import apply_lut_to_u8
    frame = (np.random.default_rng(4).random((16, 16, 3)) * 65535).astype(np.uint16)
    out = apply_lut_to_u8(frame, identity_lut(33))
    expected = (frame.astype(np.float32) / 65535.0 * 255.0 + 0.5).astype(np.uint8)
    assert np.abs(out.astype(int) - expected.astype(int)).max() <= 2


# ------------------------------------------------------------------- panels

def test_adjust_panel_maps_to_grade_params(app):
    from vct.ui.panels import AdjustPanel
    panel = AdjustPanel()
    assert panel.isNeutral()
    panel.setGradeParams(GradeParams(exposure=1.5, saturation=-0.4, tint=20))
    grade = panel.gradeParams()
    assert grade.exposure == pytest.approx(1.5, abs=0.02)
    assert grade.saturation == pytest.approx(-0.4, abs=0.02)
    assert grade.tint == pytest.approx(20, abs=1)
    assert not panel.isNeutral()
    panel.reset()
    assert panel.isNeutral()


def test_interpret_panel_applies_a_detection(app):
    from vct.ui.panels import InterpretPanel
    panel = InterpretPanel()
    detection = detect(MediaInfo(color_transfer="smpte2084", color_primaries="bt2020"))
    panel.setDetection(detection, 4000.0, "tv")
    assert panel.profileId() == "hdr10"
    assert panel.peakNits() == pytest.approx(4000.0, abs=1)
    assert panel.sourceRange() == "tv"
    # No override until the user asks for one.
    assert panel.transferOverride() is None


def test_interpret_panel_overrides(app):
    from vct.ui.panels import InterpretPanel
    panel = InterpretPanel()
    panel.advanced_check.setChecked(True)
    index = panel.transfer_combo.findData("slog3")
    panel.transfer_combo.setCurrentIndex(index)
    assert panel.transferOverride() == "slog3"


def test_interpret_panel_hides_peak_for_sdr(app):
    from vct.ui.panels import InterpretPanel
    panel = InterpretPanel()
    panel.setDetection(detect(MediaInfo(color_transfer="bt709",
                                        color_primaries="bt709")), 1000.0, "tv")
    assert not panel.peak_slider.isVisibleTo(panel)


def test_output_panel_disables_tone_mapping_for_sdr(app):
    from vct.ui.panels import OutputPanel
    panel = OutputPanel()
    panel.setHdrSource(False)
    assert not panel.mapper_combo.isEnabled()
    panel.setHdrSource(True)
    assert panel.mapper_combo.isEnabled()


def test_source_panel_flags_a_low_confidence_guess(app):
    from vct.ui.panels import SourceInfoPanel
    panel = SourceInfoPanel()
    guess = detect(MediaInfo(color_transfer="bt709", tags={"encoder": "Sony XAVC"}))
    panel.setSource(MediaInfo(path="/x/a.mov", width=1920, height=1080), guess)
    assert panel.detection_label.objectName() == "Warning"
    assert "Best guess" in panel.detection_label.text()

    confident = detect(MediaInfo(color_transfer="smpte2084", color_primaries="bt2020"))
    panel.setSource(MediaInfo(path="/x/b.mov"), confident)
    assert panel.detection_label.objectName() == "Dim"


# -------------------------------------------------------------- main window

def test_window_builds_and_defaults_to_neutral(window):
    params = window.pipelineParams()
    assert params.source.profile_id == "rec709"
    assert params.output.profile_id == "rec709"
    assert params.grade.is_neutral()
    assert not params.bypass


def test_panels_feed_the_pipeline(window):
    window.adjust_panel.setGradeParams(GradeParams(exposure=1.0, saturation=0.3))
    index = window.output_panel.mapper_combo.findData("aces")
    window.output_panel.mapper_combo.setCurrentIndex(index)
    params = window.pipelineParams()
    assert params.grade.exposure == pytest.approx(1.0, abs=0.02)
    assert params.output.tone_mapper == "aces"


def test_bypass_produces_an_identity_lut(window):
    """Bypass has to show the source untouched or it is useless for comparison."""
    window.adjust_panel.setGradeParams(GradeParams(exposure=2.0))
    window.bypass_button.setChecked(True)
    lut = build_lut(window.pipelineParams(), 17)
    assert np.abs(lut - identity_lut(17)).max() < 1e-5
    window.bypass_button.setChecked(False)
    assert np.abs(build_lut(window.pipelineParams(), 17) - identity_lut(17)).max() > 0.01


def test_draft_mode_toggles_with_interaction(window):
    window._set_draft(True)
    assert window.preview._draft
    window._set_draft(False)
    window._settle()
    assert not window.preview._draft


def test_slider_drag_stays_interactive(window):
    """The architecture's promise: a slider change costs a LUT bake, not a
    re-render of the video, so it stays interactive at any resolution."""
    index = window.interpret_panel.profile_combo.findData("hdr10")
    window.interpret_panel.profile_combo.setCurrentIndex(index)
    window._rebuild_lut()                                  # warm up

    start = time.perf_counter()
    for value in np.linspace(-2.0, 2.0, 10):
        window.adjust_panel._sliders["exposure"].setValue(float(value))
    elapsed = (time.perf_counter() - start) / 10.0
    assert elapsed < 0.075, f"{elapsed * 1000:.0f} ms per slider step"


@requires_ffmpeg
def test_opening_a_file_wires_everything_up(window, clips):
    window.openFile(clips["hdr10"])
    assert window.info is not None
    assert window.interpret_panel.profileId() == "hdr10"
    assert window.export_button.isEnabled()
    assert window.pipelineParams().source.bit_depth == 10
    window._stop_playback()


@requires_ffmpeg
def test_opening_a_junk_file_does_not_crash(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    junk = tmp_path / "notes.txt"
    junk.write_text("not a video")
    window.openFile(str(junk))
    assert window.info is None


@requires_ffmpeg
def test_saving_a_lut_from_the_window(window, clips, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from vct.core.lut import read_cube
    window.openFile(clips["hdr10"])
    window._stop_playback()
    out = str(tmp_path / "from_ui.cube")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    window.saveLut()
    lut, header = read_cube(out)
    assert header["size"] == window._lut_size
    assert np.abs(lut - identity_lut(header["size"])).max() > 0.01


@requires_ffmpeg
def test_saved_lut_ignores_bypass(window, clips, tmp_path, monkeypatch):
    """Bypass is a viewing aid; writing an identity LUT to disk would be a bug."""
    from PySide6.QtWidgets import QFileDialog
    from vct.core.lut import read_cube
    window.openFile(clips["hdr10"])
    window._stop_playback()
    window.bypass_button.setChecked(True)
    out = str(tmp_path / "bypassed.cube")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    window.saveLut()
    lut, header = read_cube(out)
    assert np.abs(lut - identity_lut(header["size"])).max() > 0.01


# ------------------------------------------------------------ export dialog

@requires_ffmpeg
def test_export_dialog_defaults_match_source(app, clips):
    from vct.core.pipeline import PipelineParams
    from vct.media.ffmpeg import detect_tools
    from vct.ui.export_dialog import ExportDialog
    info = probe(clips["sdr"])
    dialog = ExportDialog(info, PipelineParams(), detect_tools())
    settings = dialog.settings()
    assert settings.width is None and settings.height is None
    assert settings.fps is None                       # source timing preserved
    assert settings.output_path.endswith(".mp4")
    assert settings.audio_mode == "copy"
    dialog.close()


@requires_ffmpeg
def test_export_dialog_switches_container_with_codec(app, clips):
    from vct.core.pipeline import PipelineParams
    from vct.media.ffmpeg import detect_tools
    from vct.ui.export_dialog import ExportDialog
    dialog = ExportDialog(probe(clips["sdr"]), PipelineParams(), detect_tools())
    index = dialog.codec_combo.findData("prores")
    dialog.codec_combo.setCurrentIndex(index)
    settings = dialog.settings()
    assert settings.container == "mov"
    assert settings.output_path.endswith(".mov")
    assert dialog.prores_combo.isVisibleTo(dialog)
    dialog.close()


@requires_ffmpeg
def test_export_dialog_shows_the_command(app, clips):
    from vct.core.pipeline import PipelineParams
    from vct.media.ffmpeg import detect_tools
    from vct.ui.export_dialog import ExportDialog
    dialog = ExportDialog(probe(clips["sdr"]), PipelineParams(), detect_tools())
    dialog.command_toggle.setChecked(True)
    text = dialog.command_view.toPlainText()
    assert "lut3d" in text and "setparams" in text
    dialog.close()
