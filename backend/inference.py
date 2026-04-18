import io
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.utils import save_image

from ml.config import PREDICTIONS_DIR
from ml.data import denormalize
from ml.dataset_tools import split_paired_image
from ml.description import generate_description
from ml.landmarks import detect_landmarks
from ml.runtime import get_device, load_model_from_checkpoint, model_output

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")

# Process-wide model cache keyed by (model_name, checkpoint_name, device_str).
# LRU-bounded to avoid unbounded GPU memory growth. Fork-reset so child
# processes don't reuse parent's CUDA/MPS tensors (which are invalid in child).
_MODEL_CACHE_MAX = 4
_model_cache: "OrderedDict[Tuple[str, str, str], object]" = OrderedDict()
_model_cache_lock = threading.Lock()


def _reset_cache_after_fork():
    """Clear the model cache and lock in a forked child process. Tensors on
    CUDA/MPS/shared memory in the parent are not safe to use in the child.
    """
    global _model_cache, _model_cache_lock
    _model_cache = OrderedDict()
    _model_cache_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cache_after_fork)


def _get_cached_model(model_name: str, checkpoint_name: str, device: torch.device):
    """Return a cached model in eval mode, loading from disk on cache miss.

    Cache key includes device to avoid returning a CPU-resident model when
    a GPU is subsequently available (or vice-versa).
    """
    key = (model_name, checkpoint_name, str(device))
    cached = _model_cache.get(key)
    if cached is not None:
        _model_cache.move_to_end(key)  # LRU bookkeeping
        cached.eval()  # defensive: ensure eval mode even if caller toggled it
        return cached
    with _model_cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            _model_cache.move_to_end(key)
            cached.eval()
            return cached
        model, _ = load_model_from_checkpoint(model_name, checkpoint_name=checkpoint_name, device=device)
        model.eval()
        _model_cache[key] = model
        # Evict least-recently-used entries if we exceed the cap
        while len(_model_cache) > _MODEL_CACHE_MAX:
            evicted_key, _ = _model_cache.popitem(last=False)
            logger.info("Evicted model %s from cache (LRU)", evicted_key)
        logger.info("Cached model %s/%s on %s", model_name, checkpoint_name, device)
        return model

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def tensorize(image: Image.Image) -> torch.Tensor:
    return TRANSFORM(image).unsqueeze(0)


def _sanitize_filename(filename: str) -> str:
    """Strip directory components and restrict to a safe character set."""
    base = Path(filename).name  # removes path traversal components
    base = _SAFE_NAME_RE.sub("_", base)
    if not base or base in {".", ".."}:
        base = "upload"
    return base[:128]  # cap length


def save_upload(file_bytes: bytes, filename: str) -> Path:
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Upload exceeds maximum size of {MAX_UPLOAD_BYTES} bytes")
    safe = _sanitize_filename(filename)
    suffix = Path(safe).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix!r}")

    # Validate that the uploaded bytes actually decode as an image matching
    # the claimed extension. Prevents disguised non-image files from being
    # written to disk and served statically.
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(file_bytes)) as img:
            actual_format = (img.format or "").lower()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image") from exc
    ext_to_format = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
                     ".webp": "webp", ".bmp": "bmp"}
    expected = ext_to_format.get(suffix)
    if expected and actual_format != expected:
        raise ValueError(f"Image format {actual_format!r} does not match extension {suffix!r}")

    upload_dir = PREDICTIONS_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    unique = uuid.uuid4().hex[:8]
    path = upload_dir / f"{timestamp}_{unique}_{safe}"
    path.write_bytes(file_bytes)
    return path


def run_prediction(upload_path: Path, model_name: str, paired_input: bool) -> dict:
    device = get_device()
    input_image = Image.open(upload_path).convert("RGB")
    if paired_input:
        pre_image, post_image = split_paired_image(input_image)
    else:
        pre_image, post_image = input_image, None

    model = _get_cached_model(model_name, "best.pt", device)
    pre_tensor = tensorize(pre_image).to(device)
    with torch.no_grad():
        generated = model_output(model_name, model, pre_tensor)

    prediction_dir = PREDICTIONS_DIR / model_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    pre_path = prediction_dir / f"{timestamp}_pre.png"
    save_image(denormalize(pre_tensor[0]), pre_path)
    gen_path = prediction_dir / f"{timestamp}_generated.png"
    save_image(denormalize(generated[0]), gen_path)

    reference_post_path = None
    metrics = {}
    if post_image is not None:
        ref_path = prediction_dir / f"{timestamp}_reference.png"
        post_tensor = tensorize(post_image)
        save_image(denormalize(post_tensor[0]), ref_path)
        reference_post_path = ref_path
        mse = torch.nn.functional.mse_loss(generated.cpu(), post_tensor).item()
        metrics = {"paired_mse": round(mse, 6)}

    # Landmark detection on pre image
    # Cache the resized image so we don't redo the work for description generation below.
    pre_resized = pre_image.resize((256, 256))
    landmark_data = None
    try:
        lm_result = detect_landmarks(pre_resized)
        if lm_result.face_detected and lm_result.nose_features:
            landmark_data = {
                "view_type": lm_result.view_type,
                "nose_features": asdict(lm_result.nose_features),
                "nose_roi": lm_result.nose_roi,
            }
    except Exception:
        # exc_info surfaces the stack trace so silent regressions
        # (e.g. missing imports, upstream API changes) don't hide.
        logger.warning("Landmark detection failed", exc_info=True)

    # Surgery description (only if paired input with reference post)
    description_data = None
    if post_image is not None:
        try:
            post_resized = post_image.resize((256, 256))
            desc = generate_description(pre_resized, post_resized)
            if desc:
                description_data = {
                    "changes": desc.changes,
                    "summary": desc.summary,
                    "metrics": desc.detail_metrics,
                }
        except Exception:
            logger.warning("Description generation failed", exc_info=True)

    return {
        "input_mode": "paired" if paired_input else "pre_only",
        "pre_path": pre_path,
        "generated_post_path": gen_path,
        "reference_post_path": reference_post_path,
        "metrics": metrics,
        "landmarks": landmark_data,
        "description": description_data,
    }
