"""Single-window UI: drop → validate → predict → before/after + save."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QMainWindow, QMessageBox,
    QPushButton, QStatusBar, QVBoxLayout, QWidget,
)

from .core.inference_worker import InferenceRequest, InferenceWorker
from .core.paths import user_output_dir
from .core.validator import validate_image
from .widgets.before_after import BeforeAfterComparison
from .widgets.busy_overlay import BusyOverlay
from .widgets.drop_zone import DropZone
from .widgets.validation_report import ValidationReport

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main window. Single screen flow:

    1. Drag / pick photo → ``DropZone.fileDropped``
    2. Synchronous :func:`validate_image` (~100ms once InsightFace is warm)
    3. Failures: show :class:`ValidationReport`, keep predict disabled
    4. Pass: show preview, enable predict
    5. User clicks predict → :class:`InferenceWorker` runs, :class:`BusyOverlay`
       covers the window
    6. Success: show side-by-side comparison, enable Save
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CS31-1-Rhinoplasty-Prediction-Studio")
        self.resize(1020, 720)

        self._current_upload: Path | None = None
        self._current_pre: Image.Image | None = None  # loaded for inference
        self._generated: Image.Image | None = None
        self._worker: InferenceWorker | None = None

        # ----- Central layout -----
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # Top row: drop zone (left) + preview/result comparison (right).
        top = QHBoxLayout()
        top.setSpacing(16)

        self._drop = DropZone(self)
        self._drop.fileDropped.connect(self._on_file_dropped)
        top.addWidget(self._drop, stretch=1)

        # Before inference this acts as a single-image preview. After
        # inference it becomes the full-width left/right comparison panel.
        self._comparison = BeforeAfterComparison(self)
        top.addWidget(self._comparison, stretch=1)

        root.addLayout(top)

        # Inline validation report
        self._report = ValidationReport(self)
        root.addWidget(self._report)

        # Bottom actions row
        actions = QHBoxLayout()
        actions.setSpacing(12)
        actions.addStretch()

        self._change_btn = QPushButton("Choose Another Photo", self)
        self._change_btn.setObjectName("SaveButton")
        self._change_btn.setVisible(False)
        self._change_btn.setMinimumHeight(44)
        self._change_btn.clicked.connect(self._on_choose_another_clicked)
        actions.addWidget(self._change_btn)

        self._predict_btn = QPushButton("Generate Prediction", self)
        self._predict_btn.setObjectName("PredictButton")
        self._predict_btn.setEnabled(False)
        self._predict_btn.setMinimumHeight(44)
        self._predict_btn.clicked.connect(self._on_predict_clicked)
        actions.addWidget(self._predict_btn)

        self._save_btn = QPushButton("Save Result", self)
        self._save_btn.setObjectName("SaveButton")
        self._save_btn.setEnabled(False)
        self._save_btn.setMinimumHeight(44)
        self._save_btn.clicked.connect(self._on_save_clicked)
        actions.addWidget(self._save_btn)

        root.addLayout(actions)

        # ----- Busy overlay (child of central widget; sized on resize) -----
        self._busy = BusyOverlay(central)
        self._busy.cancelled.connect(self._on_cancel_clicked)
        self._busy.setGeometry(central.rect())
        self._busy.raise_()

        # ----- Status bar -----
        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._status.showMessage("Drag in a side-profile photo to start")

    # ------------------------------------------------------------------
    # Layout upkeep
    # ------------------------------------------------------------------

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        # Keep the overlay covering the central widget.
        self._busy.setGeometry(self.centralWidget().rect())

    # ------------------------------------------------------------------
    # Drop / validate
    # ------------------------------------------------------------------

    def _on_file_dropped(self, path: str) -> None:
        self._current_upload = Path(path)
        logger.info("file dropped: %s", self._current_upload)

        # Clear any previous generation.
        self._generated = None
        self._save_btn.setEnabled(False)
        self._drop.setVisible(True)
        self._change_btn.setVisible(False)

        # Load + preview first so the window is responsive during
        # validation (InsightFace first-call ~3-5s cold).
        try:
            pre_pil = Image.open(self._current_upload).convert("RGB")
        except (OSError, ValueError) as exc:
            self._report.show_errors([f"Could not read image ({exc.__class__.__name__})"])
            self._predict_btn.setEnabled(False)
            self._status.showMessage("Failed to read image")
            return

        self._current_pre = pre_pil
        # Show a single-photo preview immediately. When inference finishes,
        # the same widget switches to a side-by-side comparison.
        self._comparison.set_images(pre_pil)

        self._status.showMessage("Checking photo ...")
        self.repaint()

        result = validate_image(self._current_upload)
        if result.passed:
            self._report.clear()
            self._predict_btn.setEnabled(True)
            self._status.showMessage("Photo accepted. Click Generate.")
        else:
            self._report.show_errors(result.errors)
            self._predict_btn.setEnabled(False)
            self._status.showMessage(
                f"Requirements not met ({len(result.errors)} issue(s))"
            )

    # ------------------------------------------------------------------
    # Predict + worker lifecycle
    # ------------------------------------------------------------------

    def _on_predict_clicked(self) -> None:
        if self._current_pre is None:
            return
        base_dir = Path(os.environ["CS31_SD_BASE_DIR"])
        lora_dir = Path(os.environ["CS31_SD_LORA_DIR"])

        # Gate: SD base must be downloaded. Phase 4 will turn this into
        # an in-app onboarding flow; for now we raise a clear dialog.
        from .core.paths import is_sd_base_present
        if not is_sd_base_present():
            QMessageBox.warning(
                self, "Missing Base Model",
                "The Stable Diffusion base model (4 GB) has not been downloaded yet. "
                "For this build, please place it manually at:\n"
                f"{base_dir}\n\n"
                "The first-launch downloader can install this automatically.",
            )
            return

        request = InferenceRequest(
            pre_image=self._current_pre,
            base_dir=base_dir,
            lora_dir=lora_dir,
        )

        self._busy.reset()
        self._busy.setVisible(True)
        self._predict_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._status.showMessage("Generating ...")

        self._worker = InferenceWorker(request, parent=self)
        self._worker.progress.connect(self._busy.set_progress)
        self._worker.finished.connect(self._on_inference_finished)
        self._worker.failed.connect(self._on_inference_failed)
        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None:
            logger.info("user cancelled inference")
            self._worker.cancel()
            self._busy._hint.setText("Cancelling ...")  # noqa: SLF001 (internal UI)

    def _on_inference_finished(self, pre_pil, gen_pil) -> None:
        self._generated = gen_pil
        self._comparison.set_images(pre_pil, gen_pil)
        self._drop.setVisible(False)
        self._change_btn.setVisible(True)
        self._busy.setVisible(False)
        self._predict_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._status.showMessage("Done. Compare the original and predicted result.")
        self._worker = None

    def _on_inference_failed(self, reason: str) -> None:
        self._busy.setVisible(False)
        self._predict_btn.setEnabled(self._current_pre is not None)
        self._worker = None
        if reason == "cancelled":
            self._status.showMessage("Cancelled")
            return
        logger.warning("inference failed: %s", reason)
        self._status.showMessage(f"Failed: {reason}")
        QMessageBox.warning(self, "Generation Failed", reason)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save_clicked(self) -> None:
        if self._generated is None:
            return
        default_name = datetime.now().strftime("CS31_%Y%m%d_%H%M%S.png")
        default_dir = user_output_dir()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Generated Result",
            str(default_dir / default_name),
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if not path:
            return
        try:
            self._generated.save(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))
            return
        self._status.showMessage(f"Saved to {path}")

    def _on_choose_another_clicked(self) -> None:
        self._drop.setVisible(True)
        self._change_btn.setVisible(False)
        if not self._drop.open_file_dialog() and self._generated is not None:
            self._drop.setVisible(False)
            self._change_btn.setVisible(True)
