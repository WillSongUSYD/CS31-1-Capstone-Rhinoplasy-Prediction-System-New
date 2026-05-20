"""Side-by-side before/after comparison widget.

The generated result page uses two independent image panels instead of a
draggable split slider. Each image is aspect-fit into its own panel, so
portrait uploads keep their original proportions on screen.
"""
from __future__ import annotations

from PIL import Image

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget


def _pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL RGB image to a QPixmap via raw bytes."""

    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(
        data,
        rgba.width,
        rgba.height,
        rgba.width * 4,
        QImage.Format.Format_RGBA8888,
    )
    return QPixmap.fromImage(qimg.copy())


class BeforeAfterComparison(QWidget):
    """Render a selected photo or a left/right original-vs-generated result."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BeforeAfter")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumSize(512, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._original: QPixmap | None = None
        self._predicted: QPixmap | None = None

    def set_images(
        self,
        original: Image.Image,
        predicted: Image.Image | None = None,
    ) -> None:
        self._original = _pil_to_qpixmap(original)
        self._predicted = _pil_to_qpixmap(predicted) if predicted is not None else None
        self.update()

    def clear(self) -> None:
        self._original = None
        self._predicted = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._original is None:
            painter.setPen(QColor("#9ea9b0"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Your consultation image will appear here",
            )
            return

        if self._predicted is None:
            self._draw_single_preview(painter)
        else:
            self._draw_side_by_side(painter)

    def _draw_single_preview(self, painter: QPainter) -> None:
        assert self._original is not None
        bounds = self.rect().adjusted(24, 52, -24, -24)
        painter.setPen(QColor("#53606a"))
        painter.drawText(
            self.rect().adjusted(24, 18, -24, 0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            "Selected Photo",
        )
        self._draw_pixmap_aspect_fit(painter, self._original, bounds)

    def _draw_side_by_side(self, painter: QPainter) -> None:
        assert self._original is not None
        assert self._predicted is not None

        margin = 22
        gap = 20
        label_h = 30
        usable_w = max(1, self.width() - margin * 2 - gap)
        panel_w = usable_w // 2
        panel_h = max(1, self.height() - margin * 2)
        left_panel = QRect(margin, margin, panel_w, panel_h)
        right_panel = QRect(margin + panel_w + gap, margin, panel_w, panel_h)

        self._draw_panel(painter, left_panel, "Original", self._original, "#53606a")
        self._draw_panel(
            painter,
            right_panel,
            "Predicted Result",
            self._predicted,
            "#0d5c63",
        )

    def _draw_panel(
        self,
        painter: QPainter,
        panel: QRect,
        title: str,
        pixmap: QPixmap,
        title_color: str,
    ) -> None:
        painter.setPen(QColor(title_color))
        painter.drawText(
            panel.adjusted(0, 0, 0, 0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            title,
        )
        image_bounds = panel.adjusted(0, 34, 0, 0)
        self._draw_pixmap_aspect_fit(painter, pixmap, image_bounds)

    @staticmethod
    def _draw_pixmap_aspect_fit(
        painter: QPainter,
        pixmap: QPixmap,
        bounds: QRect,
    ) -> None:
        if pixmap.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
            return
        scale = min(bounds.width() / pixmap.width(), bounds.height() / pixmap.height())
        draw_w = max(1, int(pixmap.width() * scale))
        draw_h = max(1, int(pixmap.height() * scale))
        x = bounds.left() + (bounds.width() - draw_w) // 2
        y = bounds.top() + (bounds.height() - draw_h) // 2
        painter.drawPixmap(QRect(x, y, draw_w, draw_h), pixmap)


# Backwards-compatible name for older imports/tests.
BeforeAfterSlider = BeforeAfterComparison
