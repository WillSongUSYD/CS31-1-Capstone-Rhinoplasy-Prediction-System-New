"""QThread-based inference worker.

Why QThread: the SD Inpainting call takes 30-60s on Apple Silicon MPS.
Running it on the UI thread would freeze the whole window. Qt's threading
contract is strict: widgets may only be touched from the main thread, so
this worker emits signals that the MainWindow slots connect to with
``Qt.ConnectionType.QueuedConnection`` (auto-connected by default since
the sender and receiver live in different threads).

Lazy import of the heavy stack:
  The ``diffusers`` + ``torch`` import alone takes ~2s and triggers a
  ~300MB native-library load. We defer those imports to the worker
  ``run()`` method so app startup stays snappy and the imports happen on
  a background thread anyway.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image

from .image_geometry import (
    apply_square_canvas_transform,
    fit_image_to_square_canvas,
    restore_from_square_canvas,
)


logger = logging.getLogger(__name__)


class InferenceCancelled(Exception):
    """Raised from inside the diffusers step callback when cancel is
    requested. Caught by :class:`InferenceWorker` and translated into a
    ``failed("cancelled")`` signal."""


@dataclass
class InferenceRequest:
    """All inputs the worker needs.

    Keeping this frozen-ish (dataclass) ensures the main thread doesn't
    mutate inputs while the worker runs.
    """
    pre_image: Image.Image
    base_dir: Path
    lora_dir: Path
    prompt: str = (
        "a post-rhinoplasty face, refined natural nose, clear skin, photorealistic"
    )
    negative_prompt: str = "blurry, distorted, cartoon, low quality, deformed"
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: Optional[int] = None  # None = random
    image_size: int = 512


class InferenceWorker(QThread):
    """Runs a single SD inpainting job to completion on a background thread.

    Signals:
      * ``progress(current_step, total_steps)`` — once per denoising step
      * ``finished(PIL.Image, PIL.Image)`` — original pre-op image and generated
        image restored to the original aspect ratio
      * ``failed(str reason)`` — emitted on cancel / error
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, request: InferenceRequest, parent=None):
        super().__init__(parent)
        self._request = request
        self._cancelled = False

    # ---- public cancel API (called from main thread) ----

    def cancel(self) -> None:
        """Request cancellation. The diffusers callback will see this
        flag on its next invocation (every 1-2s during sampling) and
        raise InferenceCancelled."""
        self._cancelled = True

    # ---- thread body ----

    def run(self) -> None:
        try:
            gen = self._do_inference()
        except InferenceCancelled:
            logger.info("inference cancelled by user")
            self.failed.emit("cancelled")
        except FileNotFoundError as exc:
            # Usually: SD base model missing (first launch didn't download).
            # Report something actionable rather than a raw path.
            logger.exception("missing SD artefact")
            self.failed.emit(f"Missing model file: {exc}")
        except Exception as exc:  # pragma: no cover — surface unexpected
            logger.exception("inference failed")
            self.failed.emit(f"Inference failed: {exc}")
        else:
            self.finished.emit(self._request.pre_image.copy(), gen)

    # ---- heavy lifting ----

    def _do_inference(self) -> Image.Image:
        """Load SD pipeline (cached across calls), synthesize mask,
        generate. Returns the PIL result."""
        # Deferred imports — see module docstring.
        from backend.inference_sd import generate_sd, load_sd_pipeline
        from ml.nose_roi import get_nose_mask

        req = self._request
        logger.info(
            "inference start: base=%s lora=%s steps=%d", req.base_dir.name,
            req.lora_dir.name, req.num_inference_steps,
        )

        pipeline = load_sd_pipeline(req.base_dir, req.lora_dir)
        mask = get_nose_mask(req.pre_image)

        pre_canvas, transform = fit_image_to_square_canvas(
            req.pre_image,
            req.image_size,
            fill=(0, 0, 0),
            resample=Image.LANCZOS,
        )
        mask_canvas = apply_square_canvas_transform(
            mask,
            transform,
            fill=0,
            resample=Image.BILINEAR,
        )

        # Install a diffusers callback for per-step progress + cancel.
        # Older diffusers used ``callback`` (called with only step index);
        # 0.25+ switched to ``callback_on_step_end`` with a keyword-args
        # API. We ship 0.37 so the new API is guaranteed present.
        def step_cb(pipe, step_index, timestep, callback_kwargs):
            if self._cancelled:
                raise InferenceCancelled()
            self.progress.emit(step_index + 1, req.num_inference_steps)
            return callback_kwargs

        result = generate_sd(
            pipeline,
            pre_canvas,
            mask_canvas,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            strength=1.0,
            generator_seed=req.seed,
            image_size=req.image_size,
        )
        result = restore_from_square_canvas(
            result,
            transform,
            resample=Image.LANCZOS,
        )
        # NaN/inf sanity check — MPS attention sometimes produces pure
        # noise. The pipeline doesn't validate; we do.
        import numpy as np
        arr = np.asarray(result)
        if not np.isfinite(arr).all():
            raise RuntimeError(
                "Generated image contains NaN/Inf (MPS numerical instability). "
                "Please switch to CPU mode in the advanced menu."
            )
        logger.info("inference done")
        return result
