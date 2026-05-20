"""Single-window UI: drop → validate → predict → before/after + save."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
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

_REQUIREMENTS = [
    "One person only.",
    "Upright side profile; no frontal selfies.",
    "Nose, eyes, and mouth clearly visible.",
    "Plain white, grey, or dark background.",
    "Short side at least 512 px.",
    "Avoid hair, hands, masks, or heavy shadows.",
    "Formats: JPG, PNG, WEBP, BMP, HEIC.",
]


class MainWindow(QMainWindow):
    """Main window — header + two-column card layout.

    Flow:
    1. Drop / pick photo  → DropZone.fileDropped
    2. validate_image()   → show ValidationReport on failure
    3. Generate button    → InferenceWorker, BusyOverlay
    4. Success            → BeforeAfterComparison side-by-side
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CS31-1-Rhinoplasty-Prediction-Studio")
        self.resize(1260, 840)

        self._current_upload: Path | None = None
        self._current_pre: Image.Image | None = None
        self._generated: Image.Image | None = None
        self._worker: InferenceWorker | None = None

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 12)
        root.setSpacing(12)

        root.addWidget(self._make_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._make_left_card(), stretch=5)
        body.addWidget(self._make_right_card(), stretch=8)
        root.addLayout(body, stretch=1)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        actions.addStretch()

        self._change_btn = QPushButton("Choose Another Photo", self)
        self._change_btn.setObjectName("SaveButton")
        self._change_btn.setVisible(False)
        self._change_btn.setMinimumHeight(40)
        self._change_btn.clicked.connect(self._on_choose_another_clicked)
        actions.addWidget(self._change_btn)

        self._predict_btn = QPushButton("Generate Prediction", self)
        self._predict_btn.setObjectName("PredictButton")
        self._predict_btn.setEnabled(False)
        self._predict_btn.setMinimumHeight(40)
        self._predict_btn.clicked.connect(self._on_predict_clicked)
        actions.addWidget(self._predict_btn)

        self._save_btn = QPushButton("Save Result", self)
        self._save_btn.setObjectName("SaveButton")
        self._save_btn.setEnabled(False)
        self._save_btn.setMinimumHeight(40)
        self._save_btn.clicked.connect(self._on_save_clicked)
        actions.addWidget(self._save_btn)

        root.addLayout(actions)

        self._busy = BusyOverlay(central)
        self._busy.cancelled.connect(self._on_cancel_clicked)
        self._busy.setGeometry(central.rect())
        self._busy.raise_()

        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._status.showMessage("Drag in a side-profile photo to start")

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _make_header(self) -> QWidget:
        card = QFrame()
        card.setObjectName("HeaderCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 18, 28, 18)
        layout.setSpacing(5)

        tagline = QLabel("CS31 · RHINOPLASTY PREDICTION STUDIO")
        tagline.setObjectName("AppTagline")
        layout.addWidget(tagline)

        title = QLabel("Rhinoplasty Outcome Preview")
        title.setObjectName("AppTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Upload a clean side-profile portrait to visualize a natural, "
            "personalized nasal refinement preview while preserving the original face."
        )
        subtitle.setObjectName("AppSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        return card

    def _make_left_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("SectionCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        section_title = QLabel("Photo Intake")
        section_title.setObjectName("SectionTitle")
        layout.addWidget(section_title)

        desc = QLabel("For the most reliable preview, use a clinical-style profile photo.")
        desc.setObjectName("SectionDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self._drop = DropZone(self)
        self._drop.fileDropped.connect(self._on_file_dropped)
        layout.addWidget(self._drop, stretch=1)

        self._report = ValidationReport(self)
        layout.addWidget(self._report)

        self._req_box = self._make_requirements()
        layout.addWidget(self._req_box)

        return card

    def _make_requirements(self) -> QWidget:
        box = QFrame()
        box.setObjectName("RequirementsBox")
        box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title = QLabel("Upload Requirements")
        title.setObjectName("RequirementsTitle")
        layout.addWidget(title)

        body = QLabel("\n".join(f"- {r}" for r in _REQUIREMENTS))
        body.setObjectName("RequirementsText")
        layout.addWidget(body)

        return box

    def _make_right_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("SectionCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        section_title = QLabel("Preview Studio")
        section_title.setObjectName("SectionTitle")
        layout.addWidget(section_title)

        self._comparison = BeforeAfterComparison(self)
        layout.addWidget(self._comparison, stretch=1)

        return card

    # ------------------------------------------------------------------
    # Layout upkeep
    # ------------------------------------------------------------------

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._busy.setGeometry(self.centralWidget().rect())

    # ------------------------------------------------------------------
    # Drop / validate
    # ------------------------------------------------------------------

    def _on_file_dropped(self, path: str) -> None:
        self._current_upload = Path(path)
        logger.info("file dropped: %s", self._current_upload)

        self._generated = None
        self._save_btn.setEnabled(False)
        self._drop.setVisible(True)
        self._change_btn.setVisible(False)
        self._req_box.setVisible(True)

        try:
            pre_pil = Image.open(self._current_upload).convert("RGB")
        except (OSError, ValueError) as exc:
            self._report.show_errors([f"Could not read image ({exc.__class__.__name__})"])
            self._req_box.setVisible(False)
            self._predict_btn.setEnabled(False)
            self._status.showMessage("Failed to read image")
            return

        self._current_pre = pre_pil
        self._comparison.set_images(pre_pil)

        self._status.showMessage("Checking photo ...")
        self.repaint()

        result = validate_image(self._current_upload)
        if result.passed:
            self._report.clear()
            self._req_box.setVisible(True)
            self._predict_btn.setEnabled(True)
            self._status.showMessage("Photo accepted. Click Generate.")
        else:
            self._report.show_errors(result.errors)
            self._req_box.setVisible(False)
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
            self._busy._hint.setText("Cancelling ...")  # noqa: SLF001

    def _on_inference_finished(self, pre_pil, gen_pil) -> None:
        self._generated = gen_pil
        self._comparison.set_images(pre_pil, gen_pil)
        self._drop.setVisible(False)
        self._req_box.setVisible(False)
        self._report.clear()
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
        self._req_box.setVisible(True)
        self._change_btn.setVisible(False)
        if not self._drop.open_file_dialog() and self._generated is not None:
            self._drop.setVisible(False)
            self._req_box.setVisible(False)
            self._change_btn.setVisible(True)
