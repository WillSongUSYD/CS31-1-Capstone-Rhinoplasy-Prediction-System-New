"""First-run dialog that downloads the 4GB SD 1.5 Inpainting base model.

Flow:
  1. Dialog appears blocking launch if the base model is missing.
  2. User clicks "Download". :class:`SDBaseDownloader` runs in a QThread.
  3. Progress bar fills to 100%; dialog closes on success.
  4. On failure: show retry / cancel.

We deliberately block the main window until this completes — without
the base model the app has no function, and a silent background
download would confuse users who immediately try to use the app and
hit a "missing model" error.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QVBoxLayout,
)

from ..core.downloader import SDBaseDownloader
from ..core.paths import user_sd_base_dir


class OnboardingDialog(QDialog):
    """Modal dialog. Returns ``QDialog.Accepted`` on successful download,
    ``QDialog.Rejected`` if the user gives up."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("First Launch - Download Model")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        self._worker: SDBaseDownloader | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(14)

        title = QLabel("Download the required Stable Diffusion base model", self)
        title.setObjectName("OnboardingTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        body = QLabel(
            "The file is about 4 GB. This is a one-time download; "
            "future launches will use the local copy.\n"
            "By default, the app uses the hf-mirror.com mirror, "
            "which usually finishes in 5-10 minutes.",
            self,
        )
        body.setObjectName("OnboardingBody")
        body.setWordWrap(True)
        layout.addWidget(body)

        self._bar = QProgressBar(self)
        self._bar.setObjectName("OnboardingBar")
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("Ready to start ...")
        layout.addWidget(self._bar)

        self._detail = QLabel("", self)
        self._detail.setObjectName("OnboardingDetail")
        self._detail.setStyleSheet("color: #6d7781; font-size: 12px;")
        self._detail.setWordWrap(True)
        layout.addWidget(self._detail)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self._start_btn = QPushButton("Start Download", self)
        self._start_btn.setObjectName("OnboardingStart")
        self._start_btn.clicked.connect(self._start_download)
        buttons.addWidget(self._start_btn)
        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.setObjectName("OnboardingCancel")
        self._cancel_btn.clicked.connect(self._cancel)
        buttons.addWidget(self._cancel_btn)
        layout.addLayout(buttons)

    # ---- public API ----

    def _start_download(self) -> None:
        self._start_btn.setEnabled(False)
        self._start_btn.setText("Downloading ...")
        target = user_sd_base_dir()
        self._worker = SDBaseDownloader(target, parent=self)
        self._worker.bytes_progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._detail.setText("Cancelling ...")
        else:
            self.reject()

    # ---- signal handlers ----

    def _on_progress(self, current: int, total: int, label: str) -> None:
        if total > 0:
            pct = int(100 * current / total)
            self._bar.setValue(pct)
            mb_current = current / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self._bar.setFormat(f"{pct}%  ({mb_current:.0f} / {mb_total:.0f} MB)")
        else:
            self._bar.setFormat(f"Downloaded {current / (1024 * 1024):.0f} MB")
        # Label for the current file being fetched. HF's tqdm uses
        # human names like "(…)_pytorch_model.safetensors" so keep it
        # compact.
        if label:
            trimmed = label if len(label) < 60 else "…" + label[-57:]
            self._detail.setText(f"Current: {trimmed}")

    def _on_done(self) -> None:
        self._bar.setValue(100)
        self._bar.setFormat("Complete")
        self._detail.setText(
            "Download complete. The main window will open automatically in 2 seconds ..."
        )
        self._start_btn.setText("Continue")
        self._start_btn.setEnabled(True)
        # Rebind so "Continue" accepts the dialog instead of restarting.
        try:
            self._start_btn.clicked.disconnect()
        except TypeError:
            pass
        self._start_btn.clicked.connect(self.accept)
        self._cancel_btn.setVisible(False)
        # Auto-accept after 2s so a user who walked away doesn't return to
        # a blocked UI. They can still click "Continue" to skip the delay.
        QTimer.singleShot(2000, self.accept)

    def _on_failed(self, reason: str) -> None:
        self._start_btn.setEnabled(True)
        self._start_btn.setText("Retry")
        if reason == "cancelled":
            self._detail.setText("Cancelled")
        else:
            self._detail.setText(f"Failed: {reason}")
        self._bar.setFormat("Ready to retry")
