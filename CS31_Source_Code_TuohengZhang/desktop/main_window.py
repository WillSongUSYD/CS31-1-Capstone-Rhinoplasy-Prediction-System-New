"""Single-window UI: drop → validate → predict → before/after + save."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QFrame, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
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

# A prediction running longer than this is treated as stuck and aborted.
# A normal CPU prediction is a few minutes; 10 min is a generous ceiling.
INFERENCE_TIMEOUT_MS = 10 * 60 * 1000


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
        self.resize(1180, 760)

        self._current_upload: Path | None = None
        self._current_pre: Image.Image | None = None  # loaded for inference
        self._generated: Image.Image | None = None
        self._worker: InferenceWorker | None = None

        # Watchdog that aborts a prediction stuck past INFERENCE_TIMEOUT_MS.
        self._timed_out = False
        self._inference_timer = QTimer(self)
        self._inference_timer.setSingleShot(True)
        self._inference_timer.timeout.connect(self._on_inference_timeout)

        # ----- Central layout -----
        central = QWidget(self)
        central.setObjectName("AppShell")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 26, 28, 22)
        root.setSpacing(18)

        # Header: make the app feel like a premium consultation tool instead
        # of a plain engineering demo.
        self._hero_header = QFrame(self)
        self._hero_header.setObjectName("HeroHeader")
        header_layout = QVBoxLayout(self._hero_header)
        header_layout.setContentsMargins(24, 20, 24, 20)
        header_layout.setSpacing(6)

        eyebrow = QLabel("CS31-1-Rhinoplasty-Prediction-Studio", self)
        eyebrow.setObjectName("HeroEyebrow")
        header_layout.addWidget(eyebrow)

        title = QLabel("Rhinoplasty Outcome Preview", self)
        title.setObjectName("HeroTitle")
        header_layout.addWidget(title)

        subtitle = QLabel(
            "Upload a clean side-profile portrait to visualize a natural, "
            "personalized nasal refinement preview while preserving the original face.",
            self,
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("HeroSubtitle")
        header_layout.addWidget(subtitle)

        root.addWidget(self._hero_header)

        # Top row: drop zone (left) + preview/result comparison (right).
        top = QHBoxLayout()
        top.setSpacing(20)

        self._left_rail = QFrame(self)
        self._left_rail.setObjectName("ConsultationRail")
        self._left_rail.setMinimumWidth(390)
        self._left_rail.setMaximumWidth(470)
        left_rail_layout = QVBoxLayout(self._left_rail)
        left_rail_layout.setContentsMargins(20, 20, 20, 20)
        left_rail_layout.setSpacing(16)

        guide_title = QLabel("Photo Intake", self)
        guide_title.setObjectName("SectionTitle")
        left_rail_layout.addWidget(guide_title)

        guide_body = QLabel(
            "For the most reliable preview, use a clinical-style profile photo.",
            self,
        )
        guide_body.setWordWrap(True)
        guide_body.setObjectName("MutedBody")
        left_rail_layout.addWidget(guide_body)

        self._drop = DropZone(self)
        self._drop.fileDropped.connect(self._on_file_dropped)
        left_rail_layout.addWidget(self._drop)

        requirements = self._build_requirements_card()
        left_rail_layout.addWidget(requirements)
        left_rail_layout.addStretch()
        top.addWidget(self._left_rail, stretch=0)

        # Before inference this acts as a single-image preview. After
        # inference it becomes the full-width left/right comparison panel.
        self._result_stage = QFrame(self)
        self._result_stage.setObjectName("ResultStage")
        stage_layout = QVBoxLayout(self._result_stage)
        stage_layout.setContentsMargins(18, 18, 18, 18)
        stage_layout.setSpacing(10)

        self._stage_label = QLabel("Preview Studio", self)
        self._stage_label.setObjectName("StageTitle")
        stage_layout.addWidget(self._stage_label)

        self._comparison = BeforeAfterComparison(self)
        stage_layout.addWidget(self._comparison, stretch=1)
        top.addWidget(self._result_stage, stretch=1)

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

        self._predict_btn = QPushButton("Retry Prediction", self)
        self._predict_btn.setObjectName("PredictButton")
        self._predict_btn.setEnabled(False)
        self._predict_btn.setVisible(False)
        self._predict_btn.setMinimumHeight(44)
        self._predict_btn.clicked.connect(self._on_predict_clicked)
        actions.addWidget(self._predict_btn)

        self._save_btn = QPushButton("Save Result", self)
        self._save_btn.setObjectName("SaveButton")
        self._save_btn.setEnabled(False)
        self._save_btn.setVisible(False)
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
        self._set_intake_mode()

    def _build_requirements_card(self) -> QFrame:
        card = QFrame(self)
        card.setObjectName("RequirementsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(9)

        title = QLabel("Upload Requirements", self)
        title.setObjectName("RequirementsTitle")
        layout.addWidget(title)

        items = [
            "One person only.",
            "Upright side profile; no frontal selfies.",
            "Nose, eyes, and mouth clearly visible.",
            "Plain white, grey, or dark background.",
            "Short side at least 512 px.",
            "Avoid hair, hands, masks, or heavy shadows.",
            "Formats: JPG, PNG, WEBP, BMP, HEIC.",
        ]
        for item in items:
            row = QLabel(f"- {item}", self)
            row.setWordWrap(True)
            row.setObjectName("RequirementItem")
            layout.addWidget(row)

        return card

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
        self._set_intake_mode()

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
            self._status.showMessage("Photo accepted. Generating preview...")
            QTimer.singleShot(0, self._on_predict_clicked)
        else:
            self._report.show_errors(result.errors)
            self._predict_btn.setEnabled(False)
            self._predict_btn.setVisible(False)
            self._status.showMessage(
                f"Requirements not met ({len(result.errors)} issue(s))"
            )

    # ------------------------------------------------------------------
    # Predict + worker lifecycle
    # ------------------------------------------------------------------

    def _on_predict_clicked(self) -> None:
        if self._current_pre is None:
            return
        if self._worker is not None:
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
            self._predict_btn.setText("Retry Prediction")
            self._predict_btn.setVisible(True)
            self._predict_btn.setEnabled(True)
            return

        request = InferenceRequest(
            pre_image=self._current_pre,
            base_dir=base_dir,
            lora_dir=lora_dir,
        )

        self._busy.reset()
        self._busy.setVisible(True)
        self._set_generating_mode()
        self._predict_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._status.showMessage("Generating ...")

        self._worker = InferenceWorker(request, parent=self)
        self._worker.progress.connect(self._busy.set_progress)
        self._worker.finished.connect(self._on_inference_finished)
        self._worker.failed.connect(self._on_inference_failed)
        self._timed_out = False
        self._inference_timer.start(INFERENCE_TIMEOUT_MS)
        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None:
            logger.info("user cancelled inference")
            self._worker.cancel()
            self._busy._hint.setText("Cancelling ...")  # noqa: SLF001 (internal UI)

    def _on_inference_timeout(self) -> None:
        """Watchdog fired — the prediction has run past INFERENCE_TIMEOUT_MS.

        Request a cooperative cancel; the diffusers step callback raises
        InferenceCancelled at the next step, the worker emits failed
        ("cancelled"), and _on_inference_failed surfaces a timeout message
        because _timed_out is set.
        """
        if self._worker is not None and self._worker.isRunning():
            logger.warning(
                "prediction exceeded %d min — aborting",
                INFERENCE_TIMEOUT_MS // 60000,
            )
            self._timed_out = True
            self._worker.cancel()
            self._busy._hint.setText("Taking too long — stopping ...")  # noqa: SLF001

    def _on_inference_finished(self, pre_pil, gen_pil) -> None:
        self._inference_timer.stop()
        self._generated = gen_pil
        self._comparison.set_images(pre_pil, gen_pil)
        self._busy.setVisible(False)
        self._set_result_mode()
        self._save_btn.setEnabled(True)
        self._status.showMessage("Preview ready. Compare the original and predicted result.")
        self._worker = None

    def _on_inference_failed(self, reason: str) -> None:
        self._inference_timer.stop()
        self._busy.setVisible(False)
        self._predict_btn.setText("Retry Prediction")
        self._predict_btn.setVisible(self._current_pre is not None)
        self._predict_btn.setEnabled(self._current_pre is not None)
        self._save_btn.setEnabled(False)
        self._save_btn.setVisible(False)
        self._change_btn.setVisible(self._generated is not None)
        self._worker = None
        if reason == "cancelled":
            if self._timed_out:
                self._timed_out = False
                minutes = INFERENCE_TIMEOUT_MS // 60000
                logger.warning("inference aborted after %d-min timeout", minutes)
                self._status.showMessage(f"Stopped — prediction exceeded {minutes} minutes")
                QMessageBox.warning(
                    self, "Prediction Timed Out",
                    f"The prediction was stopped because it ran longer than "
                    f"{minutes} minutes.\n\n"
                    "This usually means the computer is too slow for CPU-based "
                    "generation, or another program is using all the CPU. "
                    "Close other heavy applications and try again.",
                )
                return
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
        self._set_intake_mode()
        if not self._drop.open_file_dialog() and self._generated is not None:
            self._set_result_mode()

    def _set_intake_mode(self) -> None:
        self._hero_header.setVisible(True)
        self._left_rail.setVisible(True)
        self._stage_label.setVisible(True)
        self._drop.setVisible(True)
        self._change_btn.setVisible(False)
        self._predict_btn.setVisible(False)
        self._save_btn.setEnabled(False)
        self._save_btn.setVisible(False)
        self._comparison.set_result_mode(False)

    def _set_generating_mode(self) -> None:
        self._hero_header.setVisible(True)
        self._left_rail.setVisible(True)
        self._stage_label.setVisible(True)
        self._drop.setVisible(True)
        self._change_btn.setVisible(False)
        self._predict_btn.setVisible(False)
        self._save_btn.setEnabled(False)
        self._save_btn.setVisible(False)
        self._comparison.set_result_mode(False)

    def _set_result_mode(self) -> None:
        self._hero_header.setVisible(False)
        self._left_rail.setVisible(False)
        self._stage_label.setVisible(False)
        self._change_btn.setVisible(True)
        self._predict_btn.setVisible(False)
        self._save_btn.setEnabled(True)
        self._save_btn.setVisible(True)
        self._comparison.set_result_mode(True)
