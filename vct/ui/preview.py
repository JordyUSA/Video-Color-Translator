"""The video preview.

Two implementations of the same idea.  Both take a decoded frame in the source's
own encoding and a 3D LUT baked from the colour pipeline, and both produce the
graded picture by sampling that LUT - the GPU with a linearly-filtered 3D
texture, the CPU with NumPy.  Neither reimplements any colour maths, which is
why the preview and the export cannot drift apart.

:func:`create_preview` returns the GL widget when a usable OpenGL context is
available and the CPU widget otherwise, so the application still runs on a
machine with no GPU driver, over remote X, or in a VM.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QSurfaceFormat
from PySide6.QtWidgets import QWidget

from ..core.lut import NEAREST, TRILINEAR, apply_lut_to_u8
from . import theme

#: Below this many pixels the CPU path can interpolate in real time; above it,
#: a drag renders a nearest-neighbour draft and refines when the drag ends.
CPU_DRAFT_PIXEL_BUDGET = 420_000


class PreviewMixin:
    """State and geometry shared by both preview implementations."""

    def _init_preview_state(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lut: Optional[np.ndarray] = None
        self._draft = False
        self._fit = True
        self._zoom = 1.0

    def setFrame(self, frame: Optional[np.ndarray]) -> None:
        self._frame = frame
        self._on_content_changed()

    def setLut(self, lut: Optional[np.ndarray]) -> None:
        self._lut = lut
        self._on_content_changed()

    def setDraft(self, draft: bool) -> None:
        """Draft mode trades interpolation quality for latency while dragging."""
        if draft != self._draft:
            self._draft = draft
            self._on_content_changed()

    @property
    def hasContent(self) -> bool:
        return self._frame is not None and self._lut is not None

    def _target_rect(self, widget_w: int, widget_h: int,
                     frame_w: int, frame_h: int) -> QRectF:
        """Largest centred rectangle preserving the source's aspect ratio."""
        if frame_w <= 0 or frame_h <= 0:
            return QRectF(0, 0, widget_w, widget_h)
        scale = min(widget_w / frame_w, widget_h / frame_h)
        width, height = frame_w * scale, frame_h * scale
        return QRectF((widget_w - width) / 2.0, (widget_h - height) / 2.0, width, height)

    def _on_content_changed(self) -> None:
        self.update()


class CpuPreview(QWidget, PreviewMixin):
    """Preview that samples the LUT with NumPy and blits a QImage.

    Always available, and fast enough to work with: a draft frame while a slider
    moves, then a trilinear render once it settles.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_preview_state()
        self._image: Optional[QImage] = None
        self._image_dirty = True
        self.setMinimumSize(320, 180)
        self.setAutoFillBackground(True)

    def _on_content_changed(self) -> None:
        self._image_dirty = True
        self.update()

    def _render(self) -> Optional[QImage]:
        if not self.hasContent:
            return None
        frame = self._frame
        interpolation = NEAREST if self._draft else TRILINEAR
        if not self._draft and frame.shape[0] * frame.shape[1] > CPU_DRAFT_PIXEL_BUDGET:
            # Big frame, settled state: still worth interpolating, just slower.
            interpolation = TRILINEAR
        rgb = apply_lut_to_u8(frame, self._lut, interpolation)
        height, width, _ = rgb.shape
        # QImage does not copy, so keep the buffer alive alongside it.
        self._buffer = np.ascontiguousarray(rgb)
        return QImage(self._buffer.data, width, height, width * 3,
                      QImage.Format_RGB888)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setPen(Qt.NoPen)
        painter.fillRect(self.rect(), theme.VIEWER_BACKGROUND)

        if self._image_dirty:
            self._image = self._render()
            self._image_dirty = False

        if self._image is None:
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Open a video file to begin")
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        target = self._target_rect(self.width(), self.height(),
                                   self._image.width(), self._image.height())
        painter.drawImage(target, self._image)


VERTEX_SHADER = """
#version 330 core
out vec2 v_uv;
void main() {
    // Fullscreen triangle: no vertex buffer needed.
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = vec2(pos.x, 1.0 - pos.y);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 v_uv;
out vec4 fragColour;

uniform sampler2D u_frame;
uniform sampler3D u_lut;
uniform float u_lutSize;
uniform vec4 u_viewport;   // xy = offset, zw = size, in normalised widget space

void main() {
    vec2 uv = (v_uv - u_viewport.xy) / u_viewport.zw;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        fragColour = vec4(0.078, 0.078, 0.078, 1.0);   // matches the surround
        return;
    }
    vec3 src = texture(u_frame, uv).rgb;
    // Half-texel correction: sample cell centres, so the table's end points land
    // exactly on 0 and 1 instead of half a cell inside them.
    vec3 coord = (clamp(src, 0.0, 1.0) * (u_lutSize - 1.0) + 0.5) / u_lutSize;
    fragColour = vec4(texture(u_lut, coord).rgb, 1.0);
}
"""


def _make_gl_preview():
    """Build the GL preview class lazily, so importing this module never
    requires the OpenGL bindings to be present."""
    from PySide6.QtOpenGL import (QOpenGLShader, QOpenGLShaderProgram,
                                  QOpenGLTexture)
    from PySide6.QtOpenGLWidgets import QOpenGLWidget

    class GlPreview(QOpenGLWidget, PreviewMixin):
        """Preview that uploads the LUT as a 3D texture and samples it in a shader.

        The frame and the LUT are separate textures updated independently, which
        is the point of the design: dragging a slider re-uploads 36 k LUT
        samples, not a multi-megapixel frame.
        """

        initialisationFailed = Signal(str)

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self._init_preview_state()
            self._program: Optional[QOpenGLShaderProgram] = None
            self._frame_texture = None
            self._lut_texture = None
            self._frame_dirty = False
            self._lut_dirty = False
            self._failed = False
            self.setMinimumSize(320, 180)

        # -- uploads -----------------------------------------------------
        def setFrame(self, frame) -> None:
            self._frame = frame
            self._frame_dirty = True
            self.update()

        def setLut(self, lut) -> None:
            self._lut = lut
            self._lut_dirty = True
            self.update()

        def setDraft(self, draft: bool) -> None:
            # The GPU interpolates for free; there is no cheaper mode to fall to.
            self._draft = draft

        def initializeGL(self) -> None:
            try:
                program = QOpenGLShaderProgram(self)
                if not program.addShaderFromSourceCode(
                        QOpenGLShader.Vertex, VERTEX_SHADER):
                    raise RuntimeError(program.log())
                if not program.addShaderFromSourceCode(
                        QOpenGLShader.Fragment, FRAGMENT_SHADER):
                    raise RuntimeError(program.log())
                if not program.link():
                    raise RuntimeError(program.log())
                self._program = program
                self._frame_dirty = self._lut_dirty = True
            except Exception as exc:                     # noqa: BLE001
                self._failed = True
                self.initialisationFailed.emit(str(exc))

        def _upload_frame(self) -> None:
            frame = self._frame
            if frame is None:
                return
            height, width, _ = frame.shape
            texture = self._frame_texture
            if (texture is None or texture.width() != width
                    or texture.height() != height):
                if texture is not None:
                    texture.destroy()
                texture = QOpenGLTexture(QOpenGLTexture.Target2D)
                texture.setFormat(QOpenGLTexture.RGB16_UNorm)
                texture.setSize(width, height)
                texture.setMipLevels(1)
                texture.allocateStorage()
                texture.setMinificationFilter(QOpenGLTexture.Linear)
                texture.setMagnificationFilter(QOpenGLTexture.Linear)
                texture.setWrapMode(QOpenGLTexture.ClampToEdge)
                self._frame_texture = texture
            data = np.ascontiguousarray(frame, dtype=np.uint16)
            texture.setData(QOpenGLTexture.RGB, QOpenGLTexture.UInt16, data.tobytes())

        def _upload_lut(self) -> None:
            lut = self._lut
            if lut is None:
                return
            size = lut.shape[0]
            texture = self._lut_texture
            if texture is None or texture.width() != size:
                if texture is not None:
                    texture.destroy()
                texture = QOpenGLTexture(QOpenGLTexture.Target3D)
                texture.setFormat(QOpenGLTexture.RGB32F)
                texture.setSize(size, size, size)
                texture.setMipLevels(1)
                texture.allocateStorage()
                texture.setMinificationFilter(QOpenGLTexture.Linear)
                texture.setMagnificationFilter(QOpenGLTexture.Linear)
                texture.setWrapMode(QOpenGLTexture.ClampToEdge)
                self._lut_texture = texture
            data = np.ascontiguousarray(lut, dtype=np.float32)
            texture.setData(QOpenGLTexture.RGB, QOpenGLTexture.Float32, data.tobytes())

        def paintGL(self) -> None:
            from PySide6.QtGui import QOpenGLContext
            functions = QOpenGLContext.currentContext().functions()
            red, green, blue = 0.078, 0.078, 0.078
            functions.glClearColor(red, green, blue, 1.0)
            functions.glClear(0x00004000)      # GL_COLOR_BUFFER_BIT

            if self._failed or self._program is None or not self.hasContent:
                return
            try:
                if self._frame_dirty:
                    self._upload_frame()
                    self._frame_dirty = False
                if self._lut_dirty:
                    self._upload_lut()
                    self._lut_dirty = False
                if self._frame_texture is None or self._lut_texture is None:
                    return

                ratio = self.devicePixelRatioF()
                width = max(int(self.width() * ratio), 1)
                height = max(int(self.height() * ratio), 1)
                rect = self._target_rect(width, height,
                                         self._frame_texture.width(),
                                         self._frame_texture.height())

                self._program.bind()
                self._frame_texture.bind(0)
                self._lut_texture.bind(1)
                self._program.setUniformValue1i("u_frame", 0)
                self._program.setUniformValue1i("u_lut", 1)
                self._program.setUniformValue1f("u_lutSize",
                                                float(self._lut.shape[0]))
                self._program.setUniformValue4f(
                    "u_viewport",
                    rect.x() / width, rect.y() / height,
                    rect.width() / width, rect.height() / height)
                functions.glDrawArrays(0x0004, 0, 3)   # GL_TRIANGLES
                self._program.release()
            except Exception as exc:                     # noqa: BLE001
                self._failed = True
                self.initialisationFailed.emit(str(exc))

    return GlPreview


def opengl_available() -> bool:
    """Whether a usable OpenGL context can be created in this session."""
    try:
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
    except ImportError:
        return False
    try:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        context = QOpenGLContext()
        context.setFormat(fmt)
        if not context.create():
            return False
        surface = QOffscreenSurface()
        surface.setFormat(fmt)
        surface.create()
        if not surface.isValid():
            return False
        ok = context.makeCurrent(surface)
        context.doneCurrent()
        return bool(ok)
    except Exception:                                    # noqa: BLE001
        return False


def create_preview(parent: Optional[QWidget] = None, force_cpu: bool = False):
    """Best preview widget this machine can run, and how it was chosen."""
    if not force_cpu and opengl_available():
        try:
            return _make_gl_preview()(parent), "opengl"
        except Exception:                                # noqa: BLE001
            pass
    return CpuPreview(parent), "cpu"


def configure_surface_format() -> None:
    """Ask for a 3.3 core context before any widget is created.

    Must run before the QApplication, which is why it lives here rather than in
    the widget: Qt bakes the default format in at that point.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)
