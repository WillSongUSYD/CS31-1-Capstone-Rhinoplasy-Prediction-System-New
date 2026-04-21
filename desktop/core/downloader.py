"""Background downloader for the SD 1.5 Inpainting base model.

Pulls only the files we actually need from HF (via the ``hf-mirror.com``
endpoint set in ``config.install_environment``). The full repo has a
redundant single-file ``.ckpt`` and ONNX exports we don't use — a naive
``snapshot_download(...)`` would waste ~6 GB of bandwidth. We filter to
the component-format files matching what the project's training pipeline
actually loaded (see V4 training notes in repo history).

Progress reporting:
    huggingface_hub exposes ``tqdm_class`` and calls our tqdm replacement
    for every chunk. We subclass tqdm and forward update()s to a Qt
    signal the UI connects to.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


# Repo mirror that serves the SD 1.5 Inpainting component format.
# The original runwayml/stable-diffusion-inpainting was removed from HF;
# botp/stable-diffusion-v1-5-inpainting is the community-maintained fork
# we verified works during V4 training.
SD_INPAINT_REPO = "botp/stable-diffusion-v1-5-inpainting"

# Allow-list picked to match V4 training-time downloads (botp repo also
# ships a legacy single-file .ckpt + ONNX exports we don't need).
SD_INPAINT_ALLOW_PATTERNS = [
    "model_index.json",
    "unet/*.bin",
    "unet/config.json",
    "vae/*.bin",
    "vae/config.json",
    "text_encoder/*.bin",
    "text_encoder/*.json",
    "tokenizer/*",
    "scheduler/*",
    "feature_extractor/*",
]


class SDBaseDownloader(QThread):
    """Download the SD Inpainting base model to ``target_dir``.

    Signals:
      * ``bytes_progress(downloaded, total, label)`` — after each chunk
        the running tqdm counters are re-read and emitted for the UI to
        render as progress bar + label ("正在下载 unet/...bin")
      * ``finished_ok()`` — emitted when snapshot_download returns
      * ``failed(reason)`` — emitted on any exception
    """

    bytes_progress = pyqtSignal(int, int, str)  # current, total, label
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, target_dir: Path, parent=None):
        super().__init__(parent)
        self._target = target_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # noqa: C901 (linear download flow)
        try:
            self._target.mkdir(parents=True, exist_ok=True)
            # Build a tqdm subclass that emits our Qt signal.
            signal = self.bytes_progress
            cancelled = lambda: self._cancelled  # noqa: E731

            import tqdm as _tqdm

            class ProgressTqdm(_tqdm.tqdm):
                """Relays tqdm state to a Qt signal on every refresh.

                HF's downloader creates one tqdm per file; we forward its
                label (``desc``) so the UI shows ``unet/...bin`` etc.
                """
                def update(self, n=1):  # type: ignore[override]
                    if cancelled():
                        # Raising from inside update stops the current
                        # tqdm's iteration. The wider snapshot_download
                        # call will re-raise as a fetch error which we
                        # catch outside.
                        raise DownloadCancelled()
                    super().update(n)
                    try:
                        signal.emit(
                            int(self.n),
                            int(self.total) if self.total else 0,
                            str(self.desc or ""),
                        )
                    except Exception:  # pragma: no cover
                        pass

            # Deferred import — huggingface_hub pulls in a few MB of
            # deps; keep them off the app launch path.
            from huggingface_hub import snapshot_download

            logger.info("starting SD base download to %s", self._target)
            snapshot_download(
                repo_id=SD_INPAINT_REPO,
                local_dir=str(self._target),
                allow_patterns=SD_INPAINT_ALLOW_PATTERNS,
                tqdm_class=ProgressTqdm,
            )
        except DownloadCancelled:
            logger.info("download cancelled by user")
            self.failed.emit("cancelled")
            return
        except Exception as exc:  # pragma: no cover — surface unexpected
            logger.exception("download failed")
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit()


class DownloadCancelled(Exception):
    """Raised by the tqdm subclass when the user cancels."""
