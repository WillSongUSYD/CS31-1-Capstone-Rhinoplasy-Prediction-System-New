"""Nose ROI extraction and paste-back for nose-only training.

Uses InsightFace 5-point landmarks + tilted-ellipse mask (matches the training
mask algorithm), so inference-time ROI cropping is consistent with how the
``nose_roi_128/`` training crops were derived.

Fallback cascade:
  1. InsightFace at det_size=320 (default)
  2. InsightFace at det_size=640 + CLAHE (for low-contrast profiles)
  3. Heuristic proportional crop anchored on non-black face region
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

from .config import ARTIFACTS_DIR, MANIFEST_PATH, SPLITS_PATH, ensure_directories

logger = logging.getLogger(__name__)

NOSE_ROI_DIR = ARTIFACTS_DIR / "dataset" / "nose_roi_128"
NOSE_ROI_SIZE = 128

# Mask algorithm constants (match ml training masks)
_MASK_CENTER_ALONG_AXIS = 0.65
_MASK_LONG_AXIS_FRAC = 0.75
_MASK_SHORT_AXIS_FRAC = 0.55
_MASK_BBOX_PAD_FRAC = 0.12  # same padding used when extracting training crops
_MASK_GAUSS_KERNEL = (25, 25)
_MASK_GAUSS_SIGMA = 9

# Fork-safe InsightFace singleton (the ONNX runtime handle is not safe to share
# across fork()-ed workers; we rebuild per-process on first use).
_app_lock = threading.Lock()
_app_instance = None
_app_pid: Optional[int] = None


def _reset_after_fork():
    global _app_instance, _app_pid, _app_lock
    _app_instance = None
    _app_pid = None
    _app_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _get_face_app(det_size: int = 320):
    """Return a per-process InsightFace FaceAnalysis singleton.

    `det_size` controls the detector input; we default to the fast 320 and
    retry at 640 only on first-pass failure.
    """
    global _app_instance, _app_pid
    pid = os.getpid()
    if _app_instance is not None and _app_pid == pid:
        return _app_instance
    with _app_lock:
        if _app_instance is not None and _app_pid == pid:
            return _app_instance
        # Inherited handle from parent -> discard, rebuild
        _app_instance = None
        try:
            from insightface.app import FaceAnalysis  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "insightface is required for nose ROI detection; pip install insightface"
            ) from exc
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        _app_instance = app
        _app_pid = pid
        return app


def _close_if_owner():
    global _app_instance, _app_pid
    if _app_instance is None or _app_pid != os.getpid():
        return
    _app_instance = None
    _app_pid = None


atexit.register(_close_if_owner)


def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """Histogram-equalise the luminance channel to help detection on low-contrast shots."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def _detect_kps(pil_image: Image.Image) -> Optional[np.ndarray]:
    """Return the 5-point landmark array or None if detection fails everywhere."""
    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Pass 1: fast
    try:
        app = _get_face_app(det_size=320)
        faces = app.get(bgr)
        if faces and faces[0].kps is not None and len(faces[0].kps) >= 3:
            return faces[0].kps
    except Exception as exc:
        logger.debug("InsightFace fast pass failed: %s", exc)

    # Pass 2: bigger det + CLAHE
    try:
        # We reuse the same singleton but the caller can accept the larger det;
        # in practice 320 vs 640 detector input both run on CPU fine.
        app = _get_face_app(det_size=320)  # keep singleton size stable
        faces = app.get(_apply_clahe(bgr))
        if faces and faces[0].kps is not None and len(faces[0].kps) >= 3:
            return faces[0].kps
    except Exception as exc:
        logger.debug("InsightFace CLAHE pass failed: %s", exc)
    return None


def _mask_from_kps(kps: np.ndarray, width: int, height: int) -> np.ndarray:
    """Build a soft mask (0-1 float32) from 5-point kps using the training geometry."""
    r_eye, l_eye, nose = kps[0], kps[1], kps[2]
    eye_mid = np.array([(r_eye[0] + l_eye[0]) / 2.0, (r_eye[1] + l_eye[1]) / 2.0], dtype=np.float32)
    nose_tip = np.array(nose, dtype=np.float32)
    axis_vec = nose_tip - eye_mid
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len < 5:
        axis_vec = np.array([0.0, 1.0], dtype=np.float32)
        axis_len = max(min(width, height) * 0.15, 20.0)
    cx, cy = eye_mid + _MASK_CENTER_ALONG_AXIS * axis_vec
    cx, cy = int(cx), int(cy)
    long_axis = max(int(axis_len * _MASK_LONG_AXIS_FRAC), 18)
    short_axis = max(int(axis_len * _MASK_SHORT_AXIS_FRAC), 14)
    angle_deg = math.degrees(math.atan2(axis_vec[1], axis_vec[0]))

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (long_axis, short_axis), angle_deg, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, _MASK_GAUSS_KERNEL, _MASK_GAUSS_SIGMA)
    mx = mask.max()
    if mx > 1e-6:
        mask = mask / mx
    return mask


def _bbox_from_mask(mask: np.ndarray, thresh: float = 0.12) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > thresh)
    if len(xs) < 4:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _pad_and_square(x1, y1, x2, y2, img_w, img_h) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * _MASK_BBOX_PAD_FRAC)
    x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
    bw = x2 - x1
    bh = y2 - y1
    if bw > bh:
        diff = bw - bh
        y1 -= diff // 2
        y2 += diff - diff // 2
    elif bh > bw:
        diff = bh - bw
        x1 -= diff // 2
        x2 += diff - diff // 2
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)
    return x1, y1, x2, y2


def _heuristic_box(w: int, h: int) -> Tuple[int, int, int, int]:
    """Last-resort proportional box, assuming face-right-facing aligned profile."""
    x1 = int(w * 0.40)
    y1 = int(h * 0.28)
    x2 = int(w * 0.88)
    y2 = int(h * 0.72)
    return x1, y1, x2, y2


def get_nose_roi_box(image: Image.Image) -> Tuple[int, int, int, int]:
    """Return the (x1, y1, x2, y2) nose crop box in pixel coordinates.

    Identical geometry to the mask used during training, so extracted crops match
    the distribution the nose-only models were trained on.
    """
    w, h = image.size
    kps = _detect_kps(image)
    if kps is not None:
        mask = _mask_from_kps(kps, w, h)
        bbox = _bbox_from_mask(mask)
        if bbox is not None:
            return _pad_and_square(*bbox, img_w=w, img_h=h)
        logger.debug("Mask produced empty bbox; falling back to heuristic.")
    else:
        logger.debug("InsightFace detection failed on image; using heuristic box.")
    return _heuristic_box(w, h)


def extract_nose_roi(image: Image.Image, target_size: int = NOSE_ROI_SIZE) -> Image.Image:
    x1, y1, x2, y2 = get_nose_roi_box(image)
    cropped = image.crop((x1, y1, x2, y2))
    return cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)


def paste_nose_back(
    original: Image.Image,
    generated_roi: Image.Image,
    blend_margin: int = 8,
) -> Image.Image:
    """Paste a generated nose ROI back onto the original face with feathered edges."""
    x1, y1, x2, y2 = get_nose_roi_box(original)
    roi_w = x2 - x1
    roi_h = y2 - y1
    resized_roi = generated_roi.resize((roi_w, roi_h), Image.Resampling.LANCZOS)

    mask = np.full((roi_h, roi_w), 255, dtype=np.float32)
    for i in range(blend_margin):
        alpha = i / blend_margin
        mask[i, :] = np.minimum(mask[i, :], alpha * 255)
        mask[-(i + 1), :] = np.minimum(mask[-(i + 1), :], alpha * 255)
        mask[:, i] = np.minimum(mask[:, i], alpha * 255)
        mask[:, -(i + 1)] = np.minimum(mask[:, -(i + 1)], alpha * 255)
    mask_pil = Image.fromarray(mask.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=blend_margin // 2)
    )

    result = original.copy()
    result.paste(resized_roi, (x1, y1), mask_pil)
    return result


def prepare_nose_rois() -> None:
    """Extract nose ROI crops for all prepared pairs and save to disk.

    Uses the InsightFace + tilted-ellipse mask bbox to keep the disk layout
    consistent with what nose-only models saw during training.
    """
    import pandas as pd
    from tqdm import tqdm

    ensure_directories()
    NOSE_ROI_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    splits = pd.read_csv(SPLITS_PATH)
    usable = manifest[
        (manifest["is_duplicate"] == False)  # noqa: E712
        & manifest["pre_path"].notna()
        & manifest["post_path"].notna()
    ]
    if "view_type" in manifest.columns:
        usable = usable[usable["view_type"] != "frontal"]
    merged = usable.merge(splits, on="sample_id", how="inner")

    count = 0
    skipped = 0
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Extracting nose ROIs"):
        pre_path = Path(row["pre_path"])
        post_path = Path(row["post_path"])
        if not pre_path.exists() or not post_path.exists():
            skipped += 1
            continue
        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")
        # Detect box on post, apply to both, so the pair shares the same ROI.
        box = get_nose_roi_box(post_img)
        for img, tag in ((pre_img, "pre"), (post_img, "post")):
            crop = img.crop(box).resize((NOSE_ROI_SIZE, NOSE_ROI_SIZE), Image.Resampling.LANCZOS)
            crop.save(NOSE_ROI_DIR / f"{row['sample_id']}_{tag}.jpg", quality=95)
        count += 1

    print(f"Extracted {count} nose ROI pairs, skipped {skipped}")


if __name__ == "__main__":
    prepare_nose_rois()
