"""Before/After comparison slider.

Two images are rendered in the same rectangle with a vertical divider;
everything LEFT of the divider is the "before" image, everything RIGHT
is the "after" (generated) image. The user drags the divider to reveal
more of either side.

Implementation is a plain ``QWidget`` with a custom ``paintEvent`` — no
``QGraphicsView`` — because we only need a single static-composite
rendering; the graphics-view machinery (transform stacks, scene
coordinate system, item-pick handling) buys us nothing for this UX.
"""
from __future__ import annotations

from PIL import Image

from PyQt6.QtCore import QPointF, QRect, Qt
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget


def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL RGB image to a QPixmap via raw bytes.

    QImage expects 32bpp RGBA8888 for the cleanest path; rather than
    introduce a PIL→QImage format handshake, we stride through
    ``QImage.Format_RGBA8888`` with an explicit alpha = 255 channel.
    Slightly wasteful on memory but avoids the per-Qt-version quirks
    around Format_RGB888 bytes-per-line padding.
    """
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, rgba.width * 4,
                  QImage.Format.Format_RGBA8888)
    # QImage points into ``data``; must copy before ``data`` goes out
    # of scope at function end.
    return QPixmap.fromImage(qimg.copy())


class BeforeAfterSlider(QWidget):
    """Widget that overlays two images with a draggable vertical divider.

    Call :meth:`set_images(pre, post)` to populate; call :meth:`clear()`
    to show the placeholder message again.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BeforeAfter")
        self.setMinimumSize(512, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._pre: QPixmap | None = None
        self._post: QPixmap | None = None
        # divider_frac is a float in [0, 1] representing the vertical
        # split position as a fraction of the drawn image rect width.
        # 0.5 = middle, 1.0 = fully "before", 0.0 = fully "after".
        self._divider_frac: float = 0.5
        self._dragging = False

    # ---- public API ----

    def set_images(self, pre: Image.Image, post: Image.Image) -> None:
        self._pre = _pil_to_qpixmap(pre)
        self._post = _pil_to_qpixmap(post)
        self._divider_frac = 0.5
        self.update()

    def clear(self) -> None:
        self._pre = None
        self._post = None
        self.update()

    # ---- painting ----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._pre is None or self._post is None:
            # Placeholder — render the same neutral box as the preview
            # pane's idle state so the layout doesn't jump when results
            # arrive.
            painter.fillRect(self.rect(), QColor("#ffffff"))
            painter.setPen(QColor("#9ea9b0"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "生成后这里会显示前后对比",
            )
            return

        # Compute the actual draw rect (aspect-fit inside widget).
        draw = self._aspect_fit_rect()

        # Before layer — draw fully, then clip and overdraw with After.
        painter.drawPixmap(draw, self._pre)

        # Divider x within the widget's coord system.
        div_x = int(draw.left() + draw.width() * self._divider_frac)

        # Clip to the "after" side (right of divider) and overlay.
        painter.save()
        clip_rect = QRect(div_x, draw.top(), draw.right() - div_x + 1, draw.height())
        painter.setClipRect(clip_rect)
        painter.drawPixmap(draw, self._post)
        painter.restore()

        # Divider line.
        pen = QPen(QColor("#ffffff"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(div_x, draw.top(), div_x, draw.bottom())

        # Handle indicator — small filled circle in the middle of the
        # divider so users understand it's draggable.
        handle_radius = 9
        handle_center = QPointF(div_x, draw.center().y())
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#0d5c63"), 2))
        painter.drawEllipse(handle_center, handle_radius, handle_radius)

        # Labels.
        painter.setPen(QColor("#ffffff"))
        painter.drawText(draw.adjusted(8, 8, 0, 0),
                         Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                         "术前")
        painter.drawText(draw.adjusted(0, 8, -8, 0),
                         Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
                         "生成术后")

    def _aspect_fit_rect(self) -> QRect:
        """Fit the images (assumed square from the pipeline) into the
        widget bounds while preserving aspect ratio."""
        if self._pre is None:
            return self.rect()
        img_w = self._pre.width()
        img_h = self._pre.height()
        w_scale = self.width() / img_w
        h_scale = self.height() / img_h
        scale = min(w_scale, h_scale)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)
        x = (self.width() - draw_w) // 2
        y = (self.height() - draw_h) // 2
        return QRect(x, y, draw_w, draw_h)

    # ---- drag handling ----

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._pre is not None:
            self._dragging = True
            self._update_divider_from_x(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            self._update_divider_from_x(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def _update_divider_from_x(self, x: float) -> None:
        draw = self._aspect_fit_rect()
        if draw.width() <= 0:
            return
        frac = (x - draw.left()) / draw.width()
        self._divider_frac = max(0.0, min(1.0, frac))
        self.update()
