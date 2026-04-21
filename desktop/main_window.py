"""Single-window UI: drop → validate → predict → before/after + save."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QStatusBar, QVBoxLayout, QWidget,
)

from .core.inference_worker import InferenceRequest, InferenceWorker
from .core.paths import user_output_dir
from .core.validator import validate_image
from .widgets.before_after import BeforeAfterSlider
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
    6. Success: swap preview for :class:`BeforeAfterSlider`, enable Save
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CS31Preview · 鼻整形术后预测")
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

        # Top row: drop zone (left) + before/after OR preview (right)
        top = QHBoxLayout()
        top.setSpacing(16)

        self._drop = DropZone(self)
        self._drop.fileDropped.connect(self._on_file_dropped)
        top.addWidget(self._drop, stretch=1)

        # We use ONE widget that doubles as preview AND slider:
        # - Before inference: acts as a QLabel-like preview (via its
        #   placeholder rendering + set_images with the same image twice
        #   so the slider shows a single image).
        # - After inference: shows pre + generated with draggable slider.
        # This avoids a layout reshuffle (preview disappearing, slider
        # appearing) which feels janky to the user.
        self._slider = BeforeAfterSlider(self)
        top.addWidget(self._slider, stretch=1)

        root.addLayout(top)

        # Inline validation report
        self._report = ValidationReport(self)
        root.addWidget(self._report)

        # Bottom actions row
        actions = QHBoxLayout()
        actions.setSpacing(12)
        actions.addStretch()

        self._predict_btn = QPushButton("生成术后图", self)
        self._predict_btn.setObjectName("PredictButton")
        self._predict_btn.setEnabled(False)
        self._predict_btn.setMinimumHeight(44)
        self._predict_btn.clicked.connect(self._on_predict_clicked)
        actions.addWidget(self._predict_btn)

        self._save_btn = QPushButton("保存结果", self)
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
        self._status.showMessage("拖入你的侧脸照片开始")

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

        # Load + preview first so the window is responsive during
        # validation (InsightFace first-call ~3-5s cold).
        try:
            pre_pil = Image.open(self._current_upload).convert("RGB")
        except (OSError, ValueError) as exc:
            self._report.show_errors([f"无法读取图片({exc.__class__.__name__})"])
            self._predict_btn.setEnabled(False)
            self._status.showMessage("读取失败")
            return

        self._current_pre = pre_pil
        # Show the uploaded image in the slider as both "before" and "after"
        # so the user sees it immediately. When inference finishes we
        # replace "after" with the generated version.
        self._slider.set_images(pre_pil, pre_pil)

        self._status.showMessage("正在检查照片 ...")
        self.repaint()

        result = validate_image(self._current_upload)
        if result.passed:
            self._report.clear()
            self._predict_btn.setEnabled(True)
            self._status.showMessage("检查通过,可点击生成")
        else:
            self._report.show_errors(result.errors)
            self._predict_btn.setEnabled(False)
            self._status.showMessage(f"不符合要求({len(result.errors)} 项问题)")

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
                self, "缺少基础模型",
                "还没下载 Stable Diffusion 基础模型(4GB)。"
                "这个版本请先手动放到:\n"
                f"{base_dir}\n\n"
                "Phase 4 会加入自动下载。",
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
        self._status.showMessage("正在生成 ...")

        self._worker = InferenceWorker(request, parent=self)
        self._worker.progress.connect(self._busy.set_progress)
        self._worker.finished.connect(self._on_inference_finished)
        self._worker.failed.connect(self._on_inference_failed)
        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None:
            logger.info("user cancelled inference")
            self._worker.cancel()
            self._busy._hint.setText("正在取消 ...")  # noqa: SLF001 (internal UI)

    def _on_inference_finished(self, pre_pil, gen_pil) -> None:
        self._generated = gen_pil
        self._slider.set_images(pre_pil, gen_pil)
        self._busy.setVisible(False)
        self._predict_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._status.showMessage("完成 · 拖动分割线对比前后")
        self._worker = None

    def _on_inference_failed(self, reason: str) -> None:
        self._busy.setVisible(False)
        self._predict_btn.setEnabled(self._current_pre is not None)
        self._worker = None
        if reason == "cancelled":
            self._status.showMessage("已取消")
            return
        logger.warning("inference failed: %s", reason)
        self._status.showMessage(f"失败:{reason}")
        QMessageBox.warning(self, "生成失败", reason)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save_clicked(self) -> None:
        if self._generated is None:
            return
        default_name = datetime.now().strftime("CS31_%Y%m%d_%H%M%S.png")
        default_dir = user_output_dir()
        path, _ = QFileDialog.getSaveFileName(
            self, "保存生成结果",
            str(default_dir / default_name),
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if not path:
            return
        try:
            self._generated.save(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._status.showMessage(f"已保存到 {path}")
