"""5-check validation pipeline for uploaded photos.

Runs BEFORE inference so a user never waits 30-60 s on SD only to be
told "that's not a side profile". Each check is cheap (<100 ms on
typical 512-2048 px input); the whole pipeline runs synchronously on
the UI thread.

Checks (in order):
  1. Resolution ≥ 512 on the short side.
  2. Exactly one face (via InsightFace ``FaceAnalysis.get``).
  3. Landmark keypoints detectable (via ``ml.nose_roi._detect_kps``).
  4. Side profile, not frontal and not basilar/tilted.
  5. Plain light background ("white wall").

Reports aggregate ALL failures in one pass so the user sees every
problem at once instead of fixing one, retrying, finding the next.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds (tunable in one place)
# ---------------------------------------------------------------------------

MIN_SHORT_SIDE_PX = 512
MAX_TILT_DEG = 30          # eye→mouth axis deviation from image-down
MIN_PROFILE_RATIO = 0.4    # |nose_dx|/eye_sep; frontal faces have ratio ~0

# Background detection:
# The V6 training data is InsightFace-aligned with BLACK padding around
# the face, not a literal "white wall". In practice, what matters to
# the model is that the background is PLAIN (uniform, no patterns/colour).
# So we accept either very bright OR very dark monochrome backgrounds.
# Either brightness extreme + low saturation → pass.
BG_MIN_VALUE_BRIGHT = 180  # ≥180 HSV V = bright (white wall)
BG_MAX_VALUE_DARK = 50     # ≤50 HSV V = dark (black backdrop / aligned crop)
BG_MAX_SATURATION = 40     # ≤40 HSV S = unsaturated (grey/white/black)
BG_PATCH_SIZE = 64         # sample patch side length in pixels


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Aggregate result of all 5 checks.

    ``errors`` holds user-facing English strings; order is stable so the
    UI can render them top-to-bottom without sorting.

    ``face_bbox`` is returned when InsightFace found a face (even if
    other checks fail), so downstream inference can reuse it without
    running detection again on the happy path.
    """
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    face_bbox: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    kps: Optional[np.ndarray] = None  # 5x2 array when available

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_image(path: Path | str) -> ValidationResult:
    """Run all 5 checks, return an aggregated ValidationResult.

    Does NOT raise on individual check failure — only on unrecoverable
    errors (file missing, unreadable). The UI needs all failure reasons
    at once.
    """
    path = Path(path)
    result = ValidationResult()

    # Load once (all checks re-use the RGB PIL + the OpenCV BGR copy).
    try:
        pil = Image.open(path).convert("RGB")
    except (OSError, ValueError) as exc:
        result.fail(f"Could not read image ({exc.__class__.__name__})")
        return result

    # Check 1 — resolution.
    w, h = pil.size
    short = min(w, h)
    if short < MIN_SHORT_SIDE_PX:
        result.fail(
            f"Photo is too small (current short side: {short}px). "
            f"Please use a clear photo with the short side at least {MIN_SHORT_SIDE_PX}px."
        )
        # Resolution failure is still recoverable for the other checks
        # (InsightFace and the HSV sampling both cope with small images),
        # so we DON'T return early - the user gets the full list.

    # Checks 2-4 require InsightFace landmarks.
    faces, kps = _detect_faces_and_kps(pil)

    # Check 2 — face count.
    if faces == 0:
        result.fail("No face detected. Try a clearer side-profile photo.")
    elif faces > 1:
        result.fail(f"Detected {faces} faces. Please upload a photo with only one person.")

    # Check 3 — landmarks (subset of check 2 but reported separately
    # because kps can fail even when face count is 1, e.g. extreme profile
    # where the det head loses the 5-point keypoint pass).
    if faces == 1 and kps is None:
        result.fail(
            "Could not detect facial landmarks. Hair, hands, or poor lighting "
            "may be blocking the face. Try another photo."
        )

    # Stash the face geometry for downstream consumers regardless of
    # whether orientation/background checks pass.
    if kps is not None:
        result.kps = kps
        result.face_bbox = _bbox_from_kps(kps, pad=0.3, img_w=w, img_h=h)

    # Check 4 — side profile. Only meaningful when we have landmarks.
    if kps is not None:
        orientation_error = _check_side_profile(kps)
        if orientation_error:
            result.fail(orientation_error)

    # Check 5 — white wall background. Runs independent of face count
    # (if we have a face bbox, we mask it out; otherwise sample corners).
    bg_error = _check_white_background(pil, result.face_bbox)
    if bg_error:
        result.fail(bg_error)

    return result


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def _detect_faces_and_kps(pil: Image.Image) -> tuple[int, Optional[np.ndarray]]:
    """Return ``(face_count, best_kps_or_None)``.

    Uses the shared InsightFace singleton from ``ml.nose_roi`` so we don't
    pay the 3-5 s model-load cost on every validation. If the cascade in
    ``_detect_kps`` can't find a face, face_count falls back to calling
    FaceAnalysis directly for the count-only semantic.
    """
    # Import here (not at module top) so importing validator.py doesn't
    # yank onnxruntime into a Qt crash-path during app startup.
    from ml.nose_roi import _detect_kps, _get_face_app

    kps = _detect_kps(pil)
    # For face count, ask the FaceAnalysis object directly. _detect_kps
    # returns only the largest face; we also need to detect "multiple
    # faces" without silently taking the biggest one.
    try:
        app = _get_face_app(det_size=640)
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        detections = app.get(bgr) or []
        face_count = len(detections)
    except Exception:  # pragma: no cover — InsightFace init failures
        logger.warning("InsightFace face-count detection failed", exc_info=True)
        face_count = 1 if kps is not None else 0

    logger.info("validation face detection: face_count=%d kps=%s",
                face_count, kps is not None)
    return face_count, kps


def _bbox_from_kps(
    kps: np.ndarray, pad: float, img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Loose bbox around the 5-point kps, padded to give the background
    check some breathing room around the face."""
    xs, ys = kps[:, 0], kps[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    bw = x2 - x1
    bh = y2 - y1
    # InsightFace kps span only the inner face (eyes/nose/mouth). Add
    # generous padding so the bbox actually covers forehead + chin.
    x1 -= bw * pad; x2 += bw * pad
    y1 -= bh * (pad + 0.8)   # extra up-padding for forehead/hair
    y2 += bh * (pad + 0.5)   # extra down for chin
    return (
        max(0, int(x1)), max(0, int(y1)),
        min(img_w, int(x2)), min(img_h, int(y2)),
    )


def _check_side_profile(kps: np.ndarray) -> Optional[str]:
    """Return error string if the face isn't an upright side profile.

    Two sub-tests, both must pass:

    * **Tilt**: eye_mid → mouth_mid axis must point roughly downward in
      image coords (|deviation| ≤ 30°). Catches basilar / upside-down /
      sideways head poses like the 0fe1ad1a... sample we removed during
      V6 dataset cleaning.
    * **Profile vs frontal**: the horizontal offset of nose_tip from
      eye_mid, normalised by inter-eye distance, must exceed
      ``MIN_PROFILE_RATIO``. Frontal faces have nose centered → ratio
      ~= 0; profiles have nose offset sideways → ratio ≥ 0.5 typically.
    """
    r_eye, l_eye, nose_tip, l_mouth, r_mouth = kps

    # Tilt: eye_mid → mouth_mid direction vs image-down (+y).
    eye_mid = (r_eye + l_eye) / 2.0
    mouth_mid = (l_mouth + r_mouth) / 2.0
    dx = mouth_mid[0] - eye_mid[0]
    dy = mouth_mid[1] - eye_mid[1]
    tilt_deg = math.degrees(math.atan2(dx, dy))
    if abs(tilt_deg) > MAX_TILT_DEG:
        return (
            f"Head pose looks unusual (tilt {abs(tilt_deg):.0f} degrees). "
            "Please use an upright side-profile photo, not an overhead, "
            "low-angle, or lying-down shot."
        )

    # Profile vs frontal: |nose_tip.x - eye_mid.x| / inter-eye distance.
    eye_sep = math.hypot(l_eye[0] - r_eye[0], l_eye[1] - r_eye[1])
    if eye_sep < 1e-3:
        # Degenerate — both "eyes" collapsed to one point. Can happen
        # when InsightFace returns a near-exact profile where it sees
        # only one eye. Treat as profile (passing).
        return None
    nose_offset = abs(nose_tip[0] - eye_mid[0])
    profile_ratio = nose_offset / eye_sep
    if profile_ratio < MIN_PROFILE_RATIO:
        return "This looks like a frontal photo. Please upload a side-profile photo."

    return None


def _check_white_background(
    pil: Image.Image, face_bbox: Optional[tuple[int, int, int, int]],
) -> Optional[str]:
    """Return error string if the background isn't plain light.

    Strategy: sample 2 patches in the TOP corners (top-left + top-right)
    of the image. Top corners are reliably outside the subject's face +
    shoulders in a typical head-and-shoulders portrait; bottom corners
    often catch clothing, which would false-positive this check.

    Pass criterion: BOTH patches must have mean V ≥ BG_MIN_VALUE and
    mean S ≤ BG_MAX_SATURATION. Both patches, not either — a photo
    taken on a colourful wall often has one corner near a window (white)
    and the other near the wall proper (colour).
    """
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    patch = BG_PATCH_SIZE
    # Clamp patch size if image is tiny (resolution check may also have
    # failed; don't crash the whole validator).
    patch = min(patch, w // 4, h // 4)
    if patch < 8:
        return None  # image too small to meaningfully sample

    # Top-left + top-right patches only (see docstring).
    patches = [
        hsv[0:patch, 0:patch],
        hsv[0:patch, w - patch:w],
    ]

    # If we know where the face is, try to avoid sampling inside it.
    # With a normal head-and-shoulders framing, the face is in the
    # middle ~1/3 of width and never touches the top corners; but for
    # tightly-cropped portraits the face CAN reach a corner. Drop any
    # patch that intersects the face bbox entirely.
    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = face_bbox
        patches = [
            p for p, (px1, py1, px2, py2) in zip(
                patches,
                [(0, 0, patch, patch), (w - patch, 0, w, patch)],
            )
            if not (px1 >= fx1 and px2 <= fx2 and py1 >= fy1 and py2 <= fy2)
        ]
    if len(patches) < 2:
        return None  # can't sample reliably; don't fail on this check

    for sample in patches:
        h_, s_, v_ = cv2.split(sample)
        mean_v = v_.mean()
        mean_s = s_.mean()
        # Background OK if it is EITHER very bright OR very dark,
        # AND it is unsaturated (greyscale). Coloured patterns fail
        # on saturation; busy scenes fail on the brightness test.
        bright_ok = mean_v >= BG_MIN_VALUE_BRIGHT and mean_s <= BG_MAX_SATURATION
        dark_ok = mean_v <= BG_MAX_VALUE_DARK and mean_s <= BG_MAX_SATURATION
        if not (bright_ok or dark_ok):
            return (
                "The background is too busy. Please take the photo against a plain wall "
                "(white is best), avoiding patterns or colorful backgrounds."
            )
    return None
