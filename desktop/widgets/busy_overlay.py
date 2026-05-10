"""Semi-transparent overlay shown while SD inference is running.

Covers the main content with a dimmed panel, a progress bar, and a
cancel button. Parent sets its geometry to cover the whole window on
``show()``.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)


class BusyOverlay(QFrame):
    """Modal-feeling overlay with a progress bar.

    Emits :attr:`cancelled` when the user clicks cancel. The owner is
    responsible for calling the actual inference-worker cancel method.
    """

    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BusyOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setVisible(False)

        panel = QWidget(self)
        panel.setObjectName("BusyPanel")
        panel.setFixedWidth(360)

        pl = QVBoxLayout(panel)
        pl.setContentsMargins(28, 24, 28, 24)
        pl.setSpacing(16)
        pl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Generating predicted after image ...", panel)
        title.setObjectName("BusyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(title)

        self._bar = QProgressBar(panel)
        self._bar.setObjectName("BusyBar")
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        pl.addWidget(self._bar)

        self._hint = QLabel(
            "Estimated 30-60 seconds (the first run may be slower while loading the 4 GB model)",
            panel,
        )
        self._hint.setObjectName("BusyHint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setWordWrap(True)
        pl.addWidget(self._hint)

        cancel = QPushButton("Cancel", panel)
        cancel.setObjectName("BusyCancel")
        cancel.clicked.connect(self.cancelled.emit)
        pl.addWidget(cancel)

        # Centre the panel.
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(panel)

    # Public API

    def set_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / max(1, total))
        self._bar.setValue(pct)
        self._bar.setFormat(f"Step {current} of {total}")

    def reset(self) -> None:
        self._bar.setValue(0)
        self._bar.setFormat("Loading model ...")
