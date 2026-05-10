"""Aspect-preserving image geometry helpers for desktop inference.

The SD inpainting model still expects a square 512x512 canvas, but user
uploads are often portrait photos. These helpers letterbox the upload into
a square canvas without stretching it, then crop the model output back to
the original aspect ratio for display and saving.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class SquareCanvasTransform:
    """Mapping between an original image and its square model canvas."""

    source_size: tuple[int, int]
    canvas_size: int
    content_box: tuple[int, int, int, int]

    @property
    def resized_size(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.content_box
        return x2 - x1, y2 - y1


def fit_image_to_square_canvas(
    image: Image.Image,
    canvas_size: int,
    *,
    fill: Any = None,
    resample: int = Image.LANCZOS,
) -> tuple[Image.Image, SquareCanvasTransform]:
    """Resize ``image`` proportionally and center it on a square canvas."""

    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive")

    scale = min(canvas_size / image.width, canvas_size / image.height)
    resized_w = max(1, min(canvas_size, round(image.width * scale)))
    resized_h = max(1, min(canvas_size, round(image.height * scale)))
    left = (canvas_size - resized_w) // 2
    top = (canvas_size - resized_h) // 2
    transform = SquareCanvasTransform(
        source_size=image.size,
        canvas_size=canvas_size,
        content_box=(left, top, left + resized_w, top + resized_h),
    )
    return apply_square_canvas_transform(
        image,
        transform,
        fill=fill,
        resample=resample,
    ), transform


def apply_square_canvas_transform(
    image: Image.Image,
    transform: SquareCanvasTransform,
    *,
    fill: Any = None,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """Apply an existing square-canvas transform to another same-size image."""

    if image.size != transform.source_size:
        raise ValueError(
            f"image size {image.size} does not match transform source size "
            f"{transform.source_size}"
        )
    x1, y1, x2, y2 = transform.content_box
    resized = image.resize((x2 - x1, y2 - y1), resample)
    canvas = Image.new(
        image.mode,
        (transform.canvas_size, transform.canvas_size),
        color=_default_fill(image, fill),
    )
    canvas.paste(resized, (x1, y1))
    return canvas


def restore_from_square_canvas(
    image: Image.Image,
    transform: SquareCanvasTransform,
    *,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """Crop out square-canvas padding and resize back to the original size."""

    if image.size != (transform.canvas_size, transform.canvas_size):
        image = image.resize((transform.canvas_size, transform.canvas_size), resample)
    crop = image.crop(transform.content_box)
    return crop.resize(transform.source_size, resample)


def _default_fill(image: Image.Image, fill: Any) -> Any:
    if fill is not None:
        return fill
    if image.mode in {"1", "L", "I", "F"}:
        return 0
    if image.mode == "RGBA":
        return (0, 0, 0, 0)
    return (0, 0, 0)
