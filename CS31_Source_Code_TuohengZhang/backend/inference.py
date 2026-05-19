import io
import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.utils import save_image

from ml import nose_roi as nose_roi_mod
from ml.config import PREDICTIONS_DIR
from ml.data import denormalize
from ml.dataset_tools import split_paired_image
from ml.description import generate_description
from ml.landmarks import detect_landmarks
from ml.runtime import get_device, load_model_from_checkpoint, model_dir, model_output

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")

# Process-wide model cache keyed by (model_name, checkpoint_name, device_str).
# LRU-bounded to avoid unbounded GPU memory growth. Fork-reset so child
# processes don't reuse parent's CUDA/MPS tensors (which are invalid in child).
#
# Locking strategy: a global lock protects the cache-dict and the
# per-key-locks dict (both are quick in-memory ops). The actual multi-second
# disk load happens under a PER-KEY lock that the global lock is NOT held
# during - so a request loading model A doesn't block a concurrent request
# loading model B. Duplicate requests for the same key still serialize
# (via the per-key lock) so we don't waste GPU memory on parallel loads
# of the same model.
_MODEL_CACHE_MAX = 4
_model_cache: "OrderedDict[Tuple[str, str, str], object]" = OrderedDict()
_model_cache_lock = threading.Lock()  # protects cache dict + per-key-locks dict
_model_loading_locks: "dict[Tuple[str, str, str], threading.Lock]" = {}


def _reset_cache_after_fork():
    """Clear the model cache and lock in a forked child process. Tensors on
    CUDA/MPS/shared memory in the parent are not safe to use in the child.
    """
    global _model_cache, _model_cache_lock, _model_loading_locks
    _model_cache = OrderedDict()
    _model_cache_lock = threading.Lock()
    _model_loading_locks = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cache_after_fork)


def _get_cached_model(model_name: str, checkpoint_name: str, device: torch.device):
    """Return a cached model in eval mode, loading from disk on cache miss.

    Cache key includes device to avoid returning a CPU-resident model when
    a GPU is subsequently available (or vice-versa).

    Concurrent callers requesting the SAME key serialize on a per-key lock
    (so we load each model only once). Concurrent callers for DIFFERENT
    keys proceed in parallel - the global lock is held only for cheap
    dict lookups, never during the multi-second disk load.
    """
    key = (model_name, checkpoint_name, str(device))
    # Fast path: already cached.
    with _model_cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            _model_cache.move_to_end(key)  # LRU bookkeeping
            cached.eval()  # defensive
            return cached
        # Miss. Grab-or-create a per-key loading lock while global lock held.
        load_lock = _model_loading_locks.setdefault(key, threading.Lock())

    # Do the load under the per-key lock only. Concurrent different-key
    # loads are now parallel (unlike the previous global-lock design).
    with load_lock:
        # Double-check: another thread with the same key may have populated
        # the cache while we waited for the per-key lock.
        with _model_cache_lock:
            cached = _model_cache.get(key)
            if cached is not None:
                _model_cache.move_to_end(key)
                cached.eval()
                return cached

        # Actual load - no locks held during disk I/O.
        model, _ = load_model_from_checkpoint(model_name, checkpoint_name=checkpoint_name, device=device)
        model.eval()

        # Insert + evict under the global lock (in-memory only).
        with _model_cache_lock:
            _model_cache[key] = model
            while len(_model_cache) > _MODEL_CACHE_MAX:
                evicted_key, _ = _model_cache.popitem(last=False)
                logger.info("Evicted model %s from cache (LRU)", evicted_key)
                # NB: we intentionally do NOT pop the per-key loading lock
                # here. Waiters blocked on the lock for `evicted_key` would
                # otherwise wake up and, finding no cache entry, grab a
                # FRESH lock from setdefault() - which lets a parallel
                # second load race against the first. Leaving the lock in
                # place bounds duplicate-load waste at the cost of a tiny
                # O(distinct-models-ever-requested) memory footprint
                # (~8 model variants in practice).
            logger.info("Cached model %s/%s on %s", model_name, checkpoint_name, device)
            return model

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Nose models were trained on a fixed-resolution nose-ROI crop (typically
# 128x128). We apply ToTensor + Normalize only; the crop is already at the
# training resolution so we deliberately do NOT Resize again.
_NOSE_POST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def tensorize(image: Image.Image) -> torch.Tensor:
    return TRANSFORM(image).unsqueeze(0)


def _is_nose_model(model_name: str) -> bool:
    return model_name.lower().endswith("_nose")


def _is_sd_inpaint_model(model_name: str) -> bool:
    """SD 1.5 Inpainting + LoRA models (V4 path C). Dispatched to a separate
    code path because the serving shape (HF pipeline in/out PIL + mask) has
    no overlap with the torch-checkpoint-based ``_get_cached_model`` flow.
    """
    name = model_name.lower()
    return name.startswith("sd_inpaint") or name.startswith("sd-inpaint")


def _sanitize_image_size(size: int, model_name: str, default: int) -> int:
    # Clamp to a reasonable range to avoid nonsensical values causing OOM
    if size < 32 or size > 1024:
        logger.warning(
            "metadata.image_size=%s for %s outside sane range; falling back to %d",
            size, model_name, default,
        )
        return default
    return size


def _read_nose_image_size(model_name: str, checkpoint_name: str = "best.pt", default: int = 128) -> int:
    """Read the training ``image_size`` for this model's checkpoint.

    Prefers the per-checkpoint ``{stem}.meta.json`` sidecar written by
    ``save_checkpoint`` (authoritative at the moment the .pt was saved),
    falling back to the whole-run ``metadata.json`` (which only reflects
    the last training run), and finally to ``default``.
    """
    directory = model_dir(model_name)
    # 1. Per-checkpoint sidecar (authoritative at save time)
    sidecar = directory / f"{Path(checkpoint_name).stem}.meta.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if "image_size" in data:
                return _sanitize_image_size(int(data["image_size"]), model_name, default)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Failed to read sidecar %s; trying metadata.json", sidecar)
    # 2. Whole-run metadata fallback
    meta_path = directory / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "image_size" in meta:
                return _sanitize_image_size(int(meta["image_size"]), model_name, default)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Failed to read metadata.json for %s; using default size %d",
                           model_name, default)
    return default


def tensorize_nose(image: Image.Image, target_size: int) -> torch.Tensor:
    """Extract the nose ROI at `target_size` and tensorize without additional resize.

    NOTE: Runs its own detection pass - prefer ``tensorize_nose_with_box`` when
    the box is already computed so extract and paste share identical coords.
    """
    crop = nose_roi_mod.extract_nose_roi(image, target_size=target_size)
    return _NOSE_POST_TRANSFORM(crop).unsqueeze(0)


def tensorize_nose_with_box(image: Image.Image, box, target_size: int) -> torch.Tensor:
    """Same as ``tensorize_nose`` but uses a pre-computed box for symmetry."""
    crop = nose_roi_mod.extract_nose_roi_with_box(image, box, target_size=target_size)
    return _NOSE_POST_TRANSFORM(crop).unsqueeze(0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a single CHW tensor in [-1, 1] space back to a PIL RGB image."""
    arr = (denormalize(tensor).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


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

    # SD Inpaint + LoRA has an entirely different serving pipeline (HF
    # pipeline, PIL in/out, on-the-fly mask synthesis). Dispatch before
    # loading a torch checkpoint to avoid load_model_from_checkpoint trying
    # to import a non-existent `sd_inpaint_nose` architecture class.
    if _is_sd_inpaint_model(model_name):
        return _run_sd_prediction(pre_image, post_image, model_name, paired_input)

    checkpoint_name = "best.pt"
    model = _get_cached_model(model_name, checkpoint_name, device)

    nose_mode = _is_nose_model(model_name)
    nose_box = None
    nose_size = None
    if nose_mode:
        # Nose models were trained on fixed-size nose-ROI crops. Use that exact
        # size; per-checkpoint sidecar records the size the checkpoint was
        # trained with (authoritative, falls back to whole-run metadata.json).
        nose_size = _read_nose_image_size(model_name, checkpoint_name=checkpoint_name, default=128)
        # Detect the ROI box ONCE and reuse for extract, paste-back, and the
        # paired reference crop. Two independent calls to get_nose_roi_box
        # can land on different InsightFace passes and silently mis-align
        # the generated paste.
        nose_box = nose_roi_mod.get_nose_roi_box(pre_image)
        pre_tensor = tensorize_nose_with_box(pre_image, nose_box, nose_size).to(device)
    else:
        pre_tensor = tensorize(pre_image).to(device)

    with torch.no_grad():
        generated = model_output(model_name, model, pre_tensor)

    prediction_dir = PREDICTIONS_DIR / model_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    pre_path = prediction_dir / f"{timestamp}_pre.png"
    save_image(denormalize(pre_tensor[0]), pre_path)
    # Copy the generated tensor to CPU exactly once and reuse it for every
    # downstream consumer (raw save, paste-back PIL conversion, description).
    # denormalize() + _tensor_to_pil() each issue their own .cpu() internally;
    # doing the copy up front avoids redundant host<->device transfers on MPS.
    gen_cpu = generated[0].detach().cpu()
    # Raw model output (for nose models this is the nose crop; for full-face
    # it's the generated post-op face). The pasted-back variant below
    # overrides gen_path for nose models so the user-facing "generated_post"
    # artifact is the full face with only the nose region replaced.
    raw_gen_path = prediction_dir / f"{timestamp}_generated_raw.png"
    save_image(denormalize(gen_cpu), raw_gen_path)
    gen_pil = _tensor_to_pil(gen_cpu)

    pasted = None
    if nose_mode:
        try:
            # Reuse the SAME box we used for extraction. This removes the
            # risk of detector drift between extract and paste landing the
            # generated nose in a different spot.
            pasted = nose_roi_mod.paste_nose_back_with_box(
                pre_image, gen_pil, nose_box,
            )
        except Exception:
            # paste_nose_back can still fail on pathological images (e.g. a
            # degenerate box with zero dimension). Fall back to the raw nose
            # crop so we still return something sensible instead of 500.
            logger.warning("paste_nose_back failed; falling back to raw nose crop",
                           exc_info=True)
            gen_path = raw_gen_path
        else:
            gen_path = prediction_dir / f"{timestamp}_generated.png"
            pasted.save(gen_path)
    else:
        gen_path = prediction_dir / f"{timestamp}_generated.png"
        save_image(denormalize(gen_cpu), gen_path)

    reference_post_path = None
    metrics = {}
    if post_image is not None:
        ref_path = prediction_dir / f"{timestamp}_reference.png"
        if nose_mode:
            # Use the SAME pre-detected box for the post crop so paired_mse
            # compares apples-to-apples. Running extract_nose_roi on post
            # would re-detect the box and land on a (potentially) different
            # crop geometry, turning the metric into noise.
            post_tensor = tensorize_nose_with_box(post_image, nose_box, nose_size)
        else:
            post_tensor = tensorize(post_image)
        save_image(denormalize(post_tensor[0]), ref_path)
        reference_post_path = ref_path
        # Force both operands onto CPU explicitly. A silent cross-device
        # reduction (e.g. generated on MPS, post_tensor on CPU) has produced
        # NaN in some torch builds; detach to drop autograd graph before move.
        mse = F.mse_loss(gen_cpu.unsqueeze(0), post_tensor.detach().cpu()).item()
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

    # Surgery description describes the MODEL'S prediction (pre -> generated),
    # not the user-provided ground-truth post. Previously we compared pre vs.
    # the user's real post image even when the API return contract said the
    # field describes the generated output - that was UX data corruption
    # (surfacing real surgical changes as if the model predicted them).
    #
    # In nose mode the raw generated tensor is a 128x128 nose-only crop with
    # no detectable face landmarks, so generate_description (which re-runs
    # landmark detection) always returned None. Use the pasted-back full face
    # as the anchor instead - it has a real face and aligned nose region.
    # When paste-back failed we have no full-face generated image to anchor
    # on, so skip description entirely rather than fabricate one.
    description_data = None
    try:
        if nose_mode:
            gen_for_desc = pasted.resize((256, 256)) if pasted is not None else None
        else:
            # Full-face mode: the generated tensor IS a full face, reuse
            # the single PIL we already built.
            gen_for_desc = gen_pil.resize((256, 256))
        if gen_for_desc is not None:
            desc = generate_description(pre_resized, gen_for_desc)
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


# ---------------------------------------------------------------------------
# SD 1.5 Inpainting + LoRA serving (V4 path C)
# ---------------------------------------------------------------------------

# Resolve SD artefact paths once. Override via env vars in deployment so
# we don't hardcode ~3.5GB of weights into the repo.
_SD_BASE_DIR_ENV = "CS31_SD_BASE_DIR"
_SD_LORA_DIR_ENV = "CS31_SD_LORA_DIR"


def _sd_artefact_paths(model_name: str) -> Tuple[Path, Path]:
    """Resolve (base_dir, lora_dir). Env-overrideable so the same binary can
    run against different base models (future SD 2.x / SDXL) without code
    changes, and against arbitrary LoRA directories for A/B testing.
    """
    repo_root = Path(__file__).resolve().parents[1]
    default_base = repo_root / "models" / "sd_base" / "inpaint"
    default_lora = repo_root / "models" / "outcome_v3_512" / model_name / "best"
    base = Path(os.environ.get(_SD_BASE_DIR_ENV, default_base))
    lora = Path(os.environ.get(_SD_LORA_DIR_ENV, default_lora))
    return base, lora


def _run_sd_prediction(
    pre_image: Image.Image,
    post_image: Optional[Image.Image],
    model_name: str,
    paired_input: bool,
) -> dict:
    """Serving path for SD 1.5 Inpainting + LoRA models.

    Contract mirrors ``run_prediction`` so the caller (serve.py route) sees
    identical response shape regardless of which model dispatched.

    Differences vs the torch-checkpoint path:
      * No ``_get_cached_model`` / ``load_model_from_checkpoint`` - SD uses a
        HF pipeline cached independently in ``backend.inference_sd``.
      * No tensorize; the pipeline consumes/produces PIL.
      * A nose mask is synthesized from InsightFace landmarks on the
        pre-op image (training data had per-sample masks; inference has to
        produce one on the fly).
      * No raw vs pasted-back split - the pipeline already outputs a full
        aligned face with the nose region regenerated, so the generated
        artefact IS the final user-facing result.
    """
    # Defer imports of heavy SD stack so the plain-torch path doesn't pay
    # diffusers/transformers import cost on every serve boot.
    from backend.inference_sd import generate_sd, load_sd_pipeline

    base_dir, lora_dir = _sd_artefact_paths(model_name)
    pipeline = load_sd_pipeline(base_dir, lora_dir)

    # Synthesize the nose mask from landmarks. We already run landmark
    # detection below for the description; running it here too is
    # acceptable (cheap vs the 30-step diffusion sample that follows).
    mask = nose_roi_mod.get_nose_mask(pre_image)

    generated = generate_sd(
        pipeline, pre_image, mask,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=1.0,
        image_size=512,
    )

    prediction_dir = PREDICTIONS_DIR / model_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    # Persist the 512x512 pre-op that actually went into the model (the
    # pipeline resized internally, but we save the same dimensions so the
    # frontend pre/gen pair is visually aligned).
    pre_512 = pre_image.resize((512, 512), Image.LANCZOS) if pre_image.size != (512, 512) else pre_image
    pre_path = prediction_dir / f"{timestamp}_pre.png"
    pre_512.save(pre_path)
    gen_path = prediction_dir / f"{timestamp}_generated.png"
    generated.save(gen_path)
    mask_path = prediction_dir / f"{timestamp}_mask.png"
    mask.save(mask_path)

    reference_post_path = None
    metrics: dict = {}
    if post_image is not None:
        # Paired-mode MSE on the mask region only. Full-face MSE is dominated
        # by unchanged background pixels and isn't a useful signal for a
        # nose-only generation task.
        ref_path = prediction_dir / f"{timestamp}_reference.png"
        post_512 = post_image.resize((512, 512), Image.LANCZOS) if post_image.size != (512, 512) else post_image
        post_512.save(ref_path)
        reference_post_path = ref_path

        gen_np = np.asarray(generated, dtype=np.float32) / 255.0
        post_np = np.asarray(post_512.convert("RGB"), dtype=np.float32) / 255.0
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0  # [H, W]
        weight = float(mask_np.sum())
        if weight > 1e-6:
            diff = (gen_np - post_np) ** 2  # [H, W, 3]
            # Broadcast mask over channels and take a mask-weighted mean.
            weighted = diff.mean(axis=-1) * mask_np
            mse = float(weighted.sum() / weight)
            metrics["paired_mse_nose"] = round(mse, 6)

    # Landmark + description on the synthesized 256 thumbnail.
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
        logger.warning("Landmark detection failed", exc_info=True)

    description_data = None
    try:
        gen_for_desc = generated.resize((256, 256))
        desc = generate_description(pre_resized, gen_for_desc)
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
