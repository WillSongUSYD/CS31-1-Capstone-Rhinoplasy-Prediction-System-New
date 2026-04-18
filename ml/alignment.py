"""Face alignment and nose ROI mask generation for profile rhinoplasty images."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _detect_skin_bbox(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect face region bounding box using skin color segmentation."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 20, 70], dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)


def align_face(pil_image: Image.Image, target_size: int = 256) -> Image.Image:
    """Align profile face: detect skin region, crop, center, normalize scale."""
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h_orig, w_orig = img_bgr.shape[:2]

    bbox = _detect_skin_bbox(img_bgr)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        # Add padding
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w_orig, x2 + pad_x)
        y2 = min(h_orig, y2 + pad_y)
        face_crop = img_bgr[y1:y2, x1:x2]
    else:
        # Fallback: use full image
        face_crop = img_bgr

    # Resize keeping aspect ratio, pad to square
    crop_h, crop_w = face_crop.shape[:2]
    scale = target_size / max(crop_w, crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized

    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def generate_nose_mask(pil_image: Image.Image, target_size: int = 256) -> np.ndarray:
    """Generate a soft elliptical nose ROI mask (0-1 float).

    Tries MediaPipe landmarks first for precision.
    Falls back to heuristic proportional mask.
    Returns a float32 array of shape (target_size, target_size) with values 0-1.
    """
    mask = np.zeros((target_size, target_size), dtype=np.float32)

    # Try MediaPipe for precise mask
    try:
        import mediapipe as mp
        from .landmarks import _create_landmarker, NOSE_TIP, NOSE_BRIDGE_TOP, LEFT_ALA, RIGHT_ALA, LEFT_NOSTRIL, RIGHT_NOSTRIL, NOSE_BRIDGE_MID

        landmarker = _create_landmarker()
        try:
            img_arr = np.array(pil_image.convert("RGB"))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_arr)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                nose_indices = [NOSE_TIP, NOSE_BRIDGE_TOP, NOSE_BRIDGE_MID, LEFT_NOSTRIL, RIGHT_NOSTRIL, LEFT_ALA, RIGHT_ALA]
                points = [(int(lm[i].x * target_size), int(lm[i].y * target_size)) for i in nose_indices]

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                rx = max(int((max(xs) - min(xs)) * 0.8), 20)
                ry = max(int((max(ys) - min(ys)) * 0.8), 20)

                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (21, 21), 7)
                mask = mask / max(mask.max(), 1e-6)
                return mask
        finally:
            landmarker.close()
    except Exception:
        pass

    # Heuristic fallback for profile views
    # Nose is typically in center-right of face, middle height
    # Detect non-black region to find actual face bounds
    img_gray = np.array(pil_image.convert("L"))
    non_black = np.where(img_gray > 15)

    if len(non_black[0]) > 100:
        y_min, y_max = non_black[0].min(), non_black[0].max()
        x_min, x_max = non_black[1].min(), non_black[1].max()
        face_w = x_max - x_min
        face_h = y_max - y_min

        # Nose center is roughly 55% from left, 42% from top of face region
        cx = x_min + int(face_w * 0.55)
        cy = y_min + int(face_h * 0.42)
        rx = max(int(face_w * 0.22), 15)
        ry = max(int(face_h * 0.18), 15)
    else:
        cx = int(target_size * 0.55)
        cy = int(target_size * 0.42)
        rx = int(target_size * 0.18)
        ry = int(target_size * 0.15)

    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 8)
    mask = mask / max(mask.max(), 1e-6)
    return mask


def visualize_mask(pil_image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Overlay green mask on image for visualization."""
    img_arr = np.array(pil_image).copy()
    green_overlay = np.zeros_like(img_arr)
    green_overlay[:, :, 1] = 255

    mask_3d = mask[:, :, None]
    blended = (img_arr * (1 - mask_3d * 0.4) + green_overlay * mask_3d * 0.4).astype(np.uint8)
    return Image.fromarray(blended)
