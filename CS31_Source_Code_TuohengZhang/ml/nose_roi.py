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
# Nose-mask ellipse geometry (tilted along eye_mid -> nose_tip axis).
#
# Tuning history:
#   V3/V4/V5 (0.65/0.75/0.55) - BROKEN: ellipse extended 10% above eye_mid
#     into the eyebrow region and 40% past the nose tip into the lips.
#     This caused the fine-tuned LoRA to co-modify eyebrows alongside the nose.
#   V6 pass 1 (0.72/0.55/0.42) - safe from eyebrows but some noses (esp.
#     profile views and long noses) had nose tip/nostrils under-covered at
#     the lower edge (1.27 × axis).
#   V6 pass 2 (0.75/0.60/0.48) - slightly larger, still safe from brows,
#     extends further to cover nose tip + nostrils + alar wings.
#
# Final range (axis units, measured from eye_mid toward nose_tip):
#   upper edge = CENTER - LONG = 0.15 (just below eye line, safe from brows)
#   lower edge = CENTER + LONG = 1.35 (past nose tip by 35%, covers columella +
#                                       nostrils + upper philtrum)
_MASK_CENTER_ALONG_AXIS = 0.75   # was 0.65
_MASK_LONG_AXIS_FRAC = 0.60      # was 0.75
_MASK_SHORT_AXIS_FRAC = 0.48     # was 0.55
_MASK_BBOX_PAD_FRAC = 0.12  # same padding used when extracting training crops
_MASK_GAUSS_KERNEL = (25, 25)
_MASK_GAUSS_SIGMA = 9

# Fork-safe InsightFace singletons keyed by det_size. The ONNX runtime handle
# is not safe to share across fork()-ed workers; we rebuild per-process on
# first use. We key by det_size so the documented pass-1 (320) and pass-2
# (640) detector sizes are actually different instances - the previous
# implementation silently ignored the det_size argument on the second call.
_app_lock = threading.Lock()
_app_instances: dict = {}  # det_size -> FaceAnalysis
_app_pid: Optional[int] = None


def _reset_after_fork():
    global _app_instances, _app_pid, _app_lock
    # Neuter inherited FaceAnalysis instances before dropping our refs:
    # Python may still hold them via GC and eventually run their finalizers,
    # which would tear down the PARENT's ONNX runtime session. Clearing
    # __dict__ removes the native-handle references so any later __del__
    # call no-ops instead of double-freeing. Mirrors the same pattern used
    # in ml/landmarks.py for MediaPipe FaceLandmarker.
    for inst in list(_app_instances.values()):
        try:
            inst.__dict__.clear()
        except Exception:
            pass
    _app_instances = {}
    _app_pid = None
    _app_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _get_face_app(det_size: int = 320):
    """Return a per-process InsightFace FaceAnalysis singleton for the given det_size.

    Separate instances are cached per det_size so pass 1 (320) and pass 2
    (640) use different prepared detectors as documented.
    """
    global _app_instances, _app_pid
    pid = os.getpid()
    if _app_pid == pid and det_size in _app_instances:
        return _app_instances[det_size]
    with _app_lock:
        if _app_pid == pid and det_size in _app_instances:
            return _app_instances[det_size]
        # Inherited handles from parent -> discard, rebuild
        if _app_pid is not None and _app_pid != pid:
            _app_instances = {}
        try:
            from insightface.app import FaceAnalysis  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "insightface is required for nose ROI detection; pip install insightface"
            ) from exc
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        _app_instances[det_size] = app
        _app_pid = pid
        return app


def _close_if_owner():
    global _app_instances, _app_pid
    if _app_pid != os.getpid():
        return
    for k in list(_app_instances.keys()):
        app = _app_instances.pop(k, None)
        # ORT doesn't expose .close() on FaceAnalysis; del is sufficient to
        # release the underlying ONNX runtime session on garbage collection.
        del app
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

    # Pass 2: bigger detector input (640) + CLAHE contrast boost. This
    # instance is cached separately from pass 1 so the prepare() call
    # actually takes effect.
    try:
        app = _get_face_app(det_size=640)
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


def _heuristic_box(w: int, h: int, bgr: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
    """Last-resort proportional box.

    Default assumes a right-facing aligned profile (face on the right half).
    When ``bgr`` is supplied, compare grayscale std-dev across left/right
    halves: the "face" side typically has more texture variation than the
    plain background side. If the left half is meaningfully more textured,
    mirror the box so it lands on the nose instead of the back of the head.
    """
    # Right-facing default.
    x1, y1, x2, y2 = int(w * 0.40), int(h * 0.28), int(w * 0.88), int(h * 0.72)
    if bgr is not None:
        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            left_std = float(gray[:, : w // 2].std())
            right_std = float(gray[:, w // 2 :].std())
            # 1.2x margin avoids flipping on near-ties (noisy backgrounds).
            if left_std > right_std * 1.2:
                x1, x2 = w - int(w * 0.88), w - int(w * 0.40)
                logger.info(
                    "Heuristic box: detected left-facing profile "
                    "(left_std=%.2f vs right_std=%.2f); mirrored box",
                    left_std, right_std,
                )
        except Exception as exc:
            # Don't let the heuristic orientation check break the
            # caller - degrade to right-facing default quietly.
            logger.debug("Heuristic orientation check failed: %s", exc)
    return x1, y1, x2, y2


def get_nose_roi_box(image: Image.Image) -> Tuple[int, int, int, int]:
    """Return the (x1, y1, x2, y2) nose crop box in pixel coordinates.

    Identical geometry to the mask used during training, so extracted crops match
    the distribution the nose-only models were trained on.
    """
    w, h = image.size
    # Keep the BGR array around so the heuristic fallback can use it for
    # a cheap left/right orientation check.
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    kps = _detect_kps(image)
    if kps is not None:
        mask = _mask_from_kps(kps, w, h)
        bbox = _bbox_from_mask(mask)
        if bbox is not None:
            return _pad_and_square(*bbox, img_w=w, img_h=h)
        logger.debug("Mask produced empty bbox; falling back to heuristic.")
    else:
        logger.debug("InsightFace detection failed on image; using heuristic box.")
    return _heuristic_box(w, h, bgr=bgr)


def get_nose_mask(image: Image.Image) -> Image.Image:
    """Return a PIL 'L' soft-edge nose mask for an image.

    Same mask geometry as the training set's ``masks_512/`` (tilted ellipse
    around the InsightFace nose axis, gaussian-softened). Needed at
    inference time for SD Inpainting - the user uploads a pre-op face but
    not a mask, so we synthesize one from landmarks.

    Fallback: when InsightFace can't detect keypoints we derive a coarse
    mask from the heuristic bounding box. That's uglier but beats refusing
    to generate entirely.

    Returns a PIL Image in mode 'L' with values 0-255 where 255 means
    "regenerate here" - matches ``StableDiffusionInpaintPipeline``'s
    mask convention.
    """
    w, h = image.size
    kps = _detect_kps(image)
    if kps is not None:
        mask_f32 = _mask_from_kps(kps, w, h)  # float32 [0, 1]
    else:
        # Fallback: rectangle from heuristic box, softened.
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = _heuristic_box(w, h, bgr=bgr)
        mask_f32 = np.zeros((h, w), dtype=np.float32)
        mask_f32[y1:y2, x1:x2] = 1.0
        mask_f32 = cv2.GaussianBlur(mask_f32, _MASK_GAUSS_KERNEL, _MASK_GAUSS_SIGMA)
        mx = mask_f32.max()
        if mx > 1e-6:
            mask_f32 = mask_f32 / mx
    mask_u8 = (mask_f32 * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mask_u8, mode="L")


def _clip_and_warn(
    box: Tuple[int, int, int, int], width: int, height: int, ctx: str = "crop",
) -> Tuple[int, int, int, int]:
    """Clip a box to image bounds, warn when clipping actually happens,
    and raise when the remaining box is empty.

    PIL's ``Image.crop`` silently pads with black on out-of-bounds coordinates
    instead of raising - that masks bugs where a box detected on one image
    size is applied to another. Explicit clipping surfaces the mismatch in
    logs and prevents silently returning a half-black crop.
    """
    x1, y1, x2, y2 = box
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(width, x2), min(height, y2)
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        logger.warning(
            "Nose box %s out of bounds for %dx%d image (%s); clipped to %s",
            box, width, height, ctx, (cx1, cy1, cx2, cy2),
        )
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(
            f"Box is empty after clipping to {width}x{height}: {(cx1, cy1, cx2, cy2)}"
        )
    return cx1, cy1, cx2, cy2


def extract_nose_roi_with_box(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    target_size: int = NOSE_ROI_SIZE,
) -> Image.Image:
    """Crop+resize a nose ROI given a pre-computed box.

    Use this when the same box must be shared across multiple operations
    (e.g. extract-generate-paste pipelines). Detecting the box independently
    at each call can pick up a different InsightFace pass outcome and cause
    silent mispasting.
    """
    w, h = image.size
    x1, y1, x2, y2 = _clip_and_warn(box, w, h, ctx="extract_nose_roi_with_box")
    cropped = image.crop((x1, y1, x2, y2))
    return cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)


def extract_nose_roi(image: Image.Image, target_size: int = NOSE_ROI_SIZE) -> Image.Image:
    """Convenience wrapper: detect box and crop in one call."""
    box = get_nose_roi_box(image)
    return extract_nose_roi_with_box(image, box, target_size=target_size)


def paste_nose_back_with_box(
    original: Image.Image,
    generated_roi: Image.Image,
    box: Tuple[int, int, int, int],
    blend_margin: int = 8,
) -> Image.Image:
    """Paste a generated nose ROI back using a pre-computed box.

    Use this when the box used to extract the pre-op ROI should be reused
    verbatim for paste-back, so there's no detector drift between the
    extract and paste steps.
    """
    w, h = original.size
    x1, y1, x2, y2 = _clip_and_warn(box, w, h, ctx="paste_nose_back_with_box")
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


def paste_nose_back(
    original: Image.Image,
    generated_roi: Image.Image,
    blend_margin: int = 8,
) -> Image.Image:
    """Convenience wrapper: detect box on ``original`` and paste in one call."""
    box = get_nose_roi_box(original)
    return paste_nose_back_with_box(original, generated_roi, box, blend_margin=blend_margin)


def prepare_nose_rois(size: int = NOSE_ROI_SIZE) -> None:
    """Extract nose ROI crops for all prepared pairs and save to disk.

    Uses the InsightFace + tilted-ellipse mask bbox to keep the disk layout
    consistent with what nose-only models saw during training. The output
    directory is named after the crop size so multiple resolutions can
    coexist (nose_roi_128, nose_roi_256, nose_roi_512, ...).
    """
    import pandas as pd
    from tqdm import tqdm

    ensure_directories()
    out_dir = ARTIFACTS_DIR / "dataset" / f"nose_roi_{size}"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=f"Extracting nose ROIs ({size}x{size})"):
        pre_path = Path(row["pre_path"])
        post_path = Path(row["post_path"])
        if not pre_path.exists() or not post_path.exists():
            skipped += 1
            continue
        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")
        # Train/infer symmetry: inference in backend/inference.py detects the
        # box on the PRE image (the only image available at inference time) and
        # reuses it for the paste-back. Keep training consistent with that so
        # the nose-only models see the same distribution in both phases.
        box = get_nose_roi_box(pre_img)
        # Paired pre/post images can differ by a pixel or two after upstream
        # alignment. PIL's Image.crop silently pads out-of-bounds regions with
        # black, which would corrupt the training set without any warning.
        # Run the box through _clip_and_warn so each image clips independently
        # (and we skip the sample entirely when the clipped box is empty).
        try:
            pre_clip = _clip_and_warn(box, *pre_img.size, ctx="prepare_nose_rois(pre)")
            post_clip = _clip_and_warn(box, *post_img.size, ctx="prepare_nose_rois(post)")
        except ValueError as exc:
            logger.warning("Skipping sample %s: %s", row["sample_id"], exc)
            skipped += 1
            continue
        for img, clip_box, tag in (
            (pre_img, pre_clip, "pre"),
            (post_img, post_clip, "post"),
        ):
            crop = img.crop(clip_box).resize((size, size), Image.Resampling.LANCZOS)
            crop.save(out_dir / f"{row['sample_id']}_{tag}.jpg", quality=95)
        count += 1

    print(f"Extracted {count} nose ROI pairs, skipped {skipped} (output: {out_dir})")


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extract nose ROI crops from prepared pairs.")
    parser.add_argument(
        "--size", type=int, default=128,
        help="Output crop size in pixels. Common values: 128 (default), 256, 512.",
    )
    args = parser.parse_args()
    prepare_nose_rois(size=args.size)


if __name__ == "__main__":
    _cli()
