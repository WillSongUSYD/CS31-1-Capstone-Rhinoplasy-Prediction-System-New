"""Facial landmark detection and nose feature extraction using MediaPipe."""

import atexit
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"

# Singleton landmarker - MediaPipe initialisation is expensive (~100ms)
# and we'd rather not pay that cost per image in a batch.
# The landmarker holds native C++ resources (GL/TFLite), so fork()-ed children
# cannot reuse a parent-owned instance. We track the owning PID and rebuild
# the instance on demand when called from a different process (e.g. DataLoader
# workers under fork/spawn). This prevents segfaults in child processes.
_landmarker_lock = threading.Lock()
_landmarker_instance = None
_landmarker_pid: Optional[int] = None
_atexit_registered = False


def _reset_after_fork():
    """Forget any inherited native state and rebuild the synchronisation lock.

    Invoked automatically in the child after fork() via os.register_at_fork().
    Called *before* the child acquires any shared state, so locks held by the
    parent cannot deadlock the child.

    The inherited FaceLandmarker instance belongs to the parent; closing it
    would double-free the native handle. Setting the module-level reference
    to None is correct but Python may still hold the inherited instance via
    GC and eventually call its ``__del__``, which internally calls native
    teardown on the parent's handle. To reduce that hazard we blank the
    inherited instance's __dict__ so the native handle reference is dropped
    before the Python wrapper's finalizer runs - worst case ``__del__`` then
    no-ops on a stripped instance instead of crashing.
    """
    global _landmarker_instance, _landmarker_pid, _landmarker_lock, _atexit_registered
    if _landmarker_instance is not None:
        try:
            _landmarker_instance.__dict__.clear()
        except Exception:
            pass
    _landmarker_instance = None  # inherited handle is unusable in child
    _landmarker_pid = None
    _landmarker_lock = threading.Lock()  # fresh lock (parent's may be held)
    _atexit_registered = False


# Register once at module import time. Available on POSIX (Linux/macOS).
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)

# MediaPipe face mesh nose-related landmark indices
NOSE_TIP = 1
NOSE_BRIDGE_TOP = 6
NOSE_BRIDGE_MID = 197
LEFT_NOSTRIL = 129
RIGHT_NOSTRIL = 358
LEFT_ALA = 49
RIGHT_ALA = 279
UPPER_LIP_TOP = 0
BETWEEN_EYES = 168
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263
CHIN = 152


@dataclass
class NoseFeatures:
    bridge_angle: float  # degrees, angle of nasal bridge relative to vertical
    tip_projection: float  # ratio: horizontal nose tip projection / nose length
    ala_width: float  # normalized ala width (distance between nostrils)
    bridge_length: float  # normalized bridge length
    nasofrontal_angle: float  # angle at bridge top (between eyes to tip)
    nasolabial_angle: float  # angle between nose base and upper lip
    symmetry_score: float  # 0-1, how symmetric the nose appears


@dataclass
class LandmarkResult:
    landmarks: List[Tuple[float, float]]  # normalized (x, y) for all 478 points
    nose_features: Optional[NoseFeatures]
    view_type: str  # "profile", "frontal", "unknown"
    nose_roi: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) pixel coords
    face_detected: bool


def _create_landmarker():
    """Create a MediaPipe FaceLandmarker instance. Prefer get_landmarker() for reuse."""
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    from mediapipe.tasks.python import BaseOptions

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face landmarker model not found at {_MODEL_PATH}. "
            "Download from https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return FaceLandmarker.create_from_options(options)


def get_landmarker():
    """Return a per-process singleton landmarker. Fork-safe: each child process
    creates its own instance on first use, because the native C++ resources
    held by FaceLandmarker cannot be shared across processes.
    """
    global _landmarker_instance, _landmarker_pid, _atexit_registered
    current_pid = os.getpid()
    # Fast path: already initialised in the current process
    if _landmarker_instance is not None and _landmarker_pid == current_pid:
        return _landmarker_instance

    with _landmarker_lock:
        # Double-checked locking with PID check: if another thread in this
        # process beat us to it, reuse their instance. Otherwise rebuild.
        if _landmarker_instance is not None and _landmarker_pid == current_pid:
            return _landmarker_instance

        # If we inherited an instance from a parent process, abandon it
        # (do NOT close - the native handle belongs to the parent).
        if _landmarker_pid is not None and _landmarker_pid != current_pid:
            logger.debug(
                "Landmarker inherited from PID %s, rebuilding for PID %s",
                _landmarker_pid, current_pid,
            )
            _landmarker_instance = None

        _landmarker_instance = _create_landmarker()
        _landmarker_pid = current_pid
        # Register atexit only once per process to avoid repeated callbacks
        if not _atexit_registered:
            atexit.register(_close_landmarker_if_owner)
            _atexit_registered = True
        return _landmarker_instance


def _close_landmarker_if_owner():
    """Close the landmarker only if we are the process that created it."""
    global _landmarker_instance, _landmarker_pid
    if _landmarker_instance is None:
        return
    if _landmarker_pid != os.getpid():
        return  # Don't close an instance we don't own
    try:
        _landmarker_instance.close()
    except Exception:
        pass
    _landmarker_instance = None
    _landmarker_pid = None


def classify_view(landmarks: list) -> str:
    """Classify face orientation as profile, frontal, or unknown.

    Returns "unknown" when the detected landmarks look degenerate (outside
    the normalized [0, 1] image range). MediaPipe occasionally emits wildly
    negative coordinates on broken/black crops - treating those as "profile"
    would poison downstream view-based dataset filtering.
    """
    nose_tip = landmarks[NOSE_TIP]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    left_eye = landmarks[LEFT_EYE_OUTER]

    # Sanity guard: landmarks are nominally in [0, 1] normalized image coords.
    # Anything outside that range usually means the detector mis-fired on a
    # degenerate crop; bail out rather than produce a confident wrong label.
    for lm in (nose_tip, right_eye, left_eye):
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return "unknown"

    eye_distance = abs(right_eye.x - left_eye.x)
    eye_center_x = (right_eye.x + left_eye.x) / 2
    nose_offset = abs(nose_tip.x - eye_center_x)

    # Collapsed-landmark guard: if the detector returned essentially the
    # same point for both eyes the result is detector junk, not a real
    # profile shot. Distinguishing it from a true profile prevents us from
    # shipping garbage through view-based filtering.
    if eye_distance < 0.005:
        return "unknown"

    # Tightened threshold from 0.08 to 0.03: 0.08 was so loose that many
    # three-quarter views were flagged profile. 0.03 corresponds to ~3% of
    # image width of inter-eye separation, which is a better proxy for a
    # true profile where one eye is nearly fully occluded.
    if eye_distance < 0.03 or (eye_distance > 0 and nose_offset / eye_distance > 0.5):
        return "profile"
    return "frontal"


def _compute_angle(p1: Tuple[float, float], vertex: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute angle at vertex in degrees."""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def extract_nose_features(landmarks: list, view_type: str) -> Optional[NoseFeatures]:
    """Extract nose measurements from face landmarks."""
    try:
        tip = (landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y)
        bridge_top = (landmarks[NOSE_BRIDGE_TOP].x, landmarks[NOSE_BRIDGE_TOP].y)
        between_eyes = (landmarks[BETWEEN_EYES].x, landmarks[BETWEEN_EYES].y)
        left_nostril = (landmarks[LEFT_NOSTRIL].x, landmarks[LEFT_NOSTRIL].y)
        right_nostril = (landmarks[RIGHT_NOSTRIL].x, landmarks[RIGHT_NOSTRIL].y)
        left_ala = (landmarks[LEFT_ALA].x, landmarks[LEFT_ALA].y)
        right_ala = (landmarks[RIGHT_ALA].x, landmarks[RIGHT_ALA].y)
        upper_lip = (landmarks[UPPER_LIP_TOP].x, landmarks[UPPER_LIP_TOP].y)

        # Bridge angle: angle of bridge line relative to vertical
        bridge_dx = tip[0] - bridge_top[0]
        bridge_dy = tip[1] - bridge_top[1]
        bridge_angle = math.degrees(math.atan2(abs(bridge_dx), abs(bridge_dy)))

        # Bridge length
        bridge_length = _dist(bridge_top, tip)

        # Tip projection: horizontal distance from bridge line to tip / bridge length
        tip_projection = abs(bridge_dx) / max(bridge_length, 1e-6)

        # Ala width
        ala_width = _dist(left_ala, right_ala)

        # Nasofrontal angle
        nasofrontal_angle = _compute_angle(between_eyes, bridge_top, tip)

        # Nasolabial angle
        nasolabial_angle = _compute_angle(bridge_top, tip, upper_lip)

        # Symmetry: compare left and right distances from nose center line
        center_x = tip[0]
        left_dist = abs(left_ala[0] - center_x)
        right_dist = abs(right_ala[0] - center_x)
        max_dist = max(left_dist, right_dist, 1e-6)
        symmetry_score = min(left_dist, right_dist) / max_dist

        return NoseFeatures(
            bridge_angle=round(bridge_angle, 2),
            tip_projection=round(tip_projection, 4),
            ala_width=round(ala_width, 4),
            bridge_length=round(bridge_length, 4),
            nasofrontal_angle=round(nasofrontal_angle, 2),
            nasolabial_angle=round(nasolabial_angle, 2),
            symmetry_score=round(symmetry_score, 4),
        )
    except (IndexError, ZeroDivisionError) as exc:
        logger.warning("Failed to extract nose features: %s", exc)
        return None


def compute_nose_roi(landmarks: list, image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
    """Compute nose region bounding box from landmarks in pixel coordinates."""
    try:
        nose_indices = [NOSE_TIP, NOSE_BRIDGE_TOP, NOSE_BRIDGE_MID, LEFT_NOSTRIL, RIGHT_NOSTRIL, LEFT_ALA, RIGHT_ALA]
        xs = [landmarks[i].x * image_width for i in nose_indices]
        ys = [landmarks[i].y * image_height for i in nose_indices]

        # Add padding (20% of nose region size)
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        pad_x = w * 0.3
        pad_y = h * 0.3

        x1 = max(0, int(min(xs) - pad_x))
        y1 = max(0, int(min(ys) - pad_y))
        x2 = min(image_width, int(max(xs) + pad_x))
        y2 = min(image_height, int(max(ys) + pad_y))

        return (x1, y1, x2, y2)
    except (IndexError, ValueError):
        return None


def detect_landmarks(image: Image.Image) -> LandmarkResult:
    """Detect face landmarks and extract nose features from a PIL image."""
    import mediapipe as mp

    landmarker = get_landmarker()
    img_arr = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_arr)
    # MediaPipe FaceLandmarker is not documented as thread-safe; the singleton
    # pattern above means concurrent FastAPI requests would otherwise issue
    # overlapping detect() calls into one native instance. Serialize the call
    # under the same lock used for lazy-init.
    with _landmarker_lock:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return LandmarkResult(
            landmarks=[], nose_features=None, view_type="unknown",
            nose_roi=None, face_detected=False,
        )

    lm = result.face_landmarks[0]
    all_points = [(l.x, l.y) for l in lm]
    view_type = classify_view(lm)
    nose_features = extract_nose_features(lm, view_type)
    nose_roi = compute_nose_roi(lm, image.width, image.height)

    return LandmarkResult(
        landmarks=all_points, nose_features=nose_features,
        view_type=view_type, nose_roi=nose_roi, face_detected=True,
    )


def detect_view_type(image: Image.Image) -> str:
    """Quick helper: detect only the view type without full feature extraction."""
    import mediapipe as mp

    landmarker = get_landmarker()
    img_arr = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_arr)
    # See detect_landmarks() for the thread-safety rationale.
    with _landmarker_lock:
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return "unknown"
    return classify_view(result.face_landmarks[0])


def batch_detect_view_types(images: List[Image.Image]) -> List[str]:
    """Detect view types for a batch of images using the singleton landmarker."""
    import mediapipe as mp

    landmarker = get_landmarker()
    results = []
    for image in images:
        img_arr = np.array(image.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_arr)
        # Serialize each detect() call (see detect_landmarks). Holding the
        # lock per-image rather than for the whole loop keeps other callers
        # interleavable - a long batch otherwise starves concurrent requests.
        with _landmarker_lock:
            result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            results.append("unknown")
        else:
            results.append(classify_view(result.face_landmarks[0]))
    return results


def draw_landmarks_on_image(image: Image.Image, landmark_result: LandmarkResult) -> Image.Image:
    """Draw nose landmarks and ROI on a copy of the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    if not landmark_result.face_detected:
        return img

    w, h = img.size
    # Draw nose landmarks
    nose_indices = [NOSE_TIP, NOSE_BRIDGE_TOP, NOSE_BRIDGE_MID, LEFT_NOSTRIL, RIGHT_NOSTRIL, LEFT_ALA, RIGHT_ALA]
    for idx in nose_indices:
        if idx < len(landmark_result.landmarks):
            x, y = landmark_result.landmarks[idx]
            px, py = int(x * w), int(y * h)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(0, 255, 0))

    # Draw nose ROI
    if landmark_result.nose_roi:
        x1, y1, x2, y2 = landmark_result.nose_roi
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)

    return img
