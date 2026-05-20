"""DropZone — drag-drop + file-picker entry widget.

Emits ``fileDropped(str path)`` when the user drops a single image file or
picks one through the native dialog.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout


_ACCEPTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}


class DropZone(QFrame):
    """Large dashed-border rectangle that accepts a single image drop.

    Keeps UI-state self-contained: hover highlight, invalid-file rejection,
    "click anywhere" file-picker fallback. All higher-level logic (5-check
    validation, inference) happens outside, reacting to the
    ``fileDropped(str)`` signal.
    """

    fileDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(240)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel("Drop a profile portrait\nor click to select a photo", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setObjectName("DropZoneLabel")
        layout.addWidget(self._label)

        # Initial neutral style. Active/invalid styles get toggled via QSS
        # property selectors (see assets/style.qss).
        self.setProperty("dropState", "idle")

    # ---- click-to-open fallback (for users who don't drag) ----

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_file_dialog()
        super().mousePressEvent(event)

    def open_file_dialog(self) -> bool:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose your side-profile photo",
            str(Path.home() / "Pictures"),
            "Image files (*.jpg *.jpeg *.png *.webp *.bmp *.heic)",
        )
        if path:
            self.fileDropped.emit(path)
            return True
        return False

    # ---- drag-drop handling ----

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Reject multi-file drops as early as possible so the user sees
        # the "no" cursor instead of getting a late popup.
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) == 1 and self._is_acceptable(urls[0].toLocalFile()):
            event.acceptProposedAction()
            self._set_state("active")
        else:
            event.ignore()
            self._set_state("invalid")

    def dragLeaveEvent(self, event):
        self._set_state("idle")
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        self._set_state("idle")
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if len(urls) != 1:
            return  # shouldn't happen; dragEnter already rejected
        path = urls[0].toLocalFile()
        if self._is_acceptable(path):
            self.fileDropped.emit(path)
            event.acceptProposedAction()

    # ---- helpers ----

    @staticmethod
    def _is_acceptable(path: str) -> bool:
        return Path(path).suffix.lower() in _ACCEPTED_SUFFIXES

    def _set_state(self, state: str):
        """Toggle the ``dropState`` property; trigger QSS re-polish."""
        self.setProperty("dropState", state)
        self.style().unpolish(self)
        self.style().polish(self)
