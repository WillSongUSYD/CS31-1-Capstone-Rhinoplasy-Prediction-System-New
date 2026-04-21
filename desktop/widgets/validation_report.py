"""Inline report panel for validation failures.

Sits between the drop zone and the predict button. Empty/invisible when
validation passes or no file is loaded; red banner with bullet list of
issues when any check fails.

A dedicated widget (rather than QMessageBox) so the user can read all
five reasons at once without dismissing a modal, and edit their upload
in parallel (drag a new photo → report refreshes).
"""
from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class ValidationReport(QFrame):
    """Red-background list of validation errors.

    Call :meth:`show_errors` to populate, :meth:`clear` to hide.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ValidationReport")
        self.setVisible(False)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 12, 16, 12)
        self._layout.setSpacing(4)

        self._title = QLabel("这张照片不符合要求:", self)
        self._title.setObjectName("ValidationReportTitle")
        self._layout.addWidget(self._title)

        self._bullets: list[QLabel] = []

    def show_errors(self, errors: list[str]) -> None:
        """Replace any previous bullets with the given error list."""
        # Drop previous bullet labels; keep the title.
        for b in self._bullets:
            self._layout.removeWidget(b)
            b.deleteLater()
        self._bullets.clear()

        for msg in errors:
            bullet = QLabel(f"•  {msg}", self)
            bullet.setWordWrap(True)
            bullet.setObjectName("ValidationBullet")
            self._layout.addWidget(bullet)
            self._bullets.append(bullet)

        self.setVisible(bool(errors))

    def clear(self) -> None:
        for b in self._bullets:
            self._layout.removeWidget(b)
            b.deleteLater()
        self._bullets.clear()
        self.setVisible(False)
