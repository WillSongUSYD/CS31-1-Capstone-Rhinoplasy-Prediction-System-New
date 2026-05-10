from PIL import Image

from desktop.core.image_geometry import (
    apply_square_canvas_transform,
    fit_image_to_square_canvas,
    restore_from_square_canvas,
)


def test_portrait_image_letterbox_restores_original_size():
    image = Image.new("RGB", (900, 1600), color=(255, 0, 0))

    canvas, transform = fit_image_to_square_canvas(image, 512, fill=(0, 0, 0))
    restored = restore_from_square_canvas(canvas, transform)

    assert canvas.size == (512, 512)
    assert transform.resized_size == (288, 512)
    assert restored.size == image.size
    assert restored.getpixel((0, 0)) == (255, 0, 0)


def test_square_image_has_no_padding_and_restores_original_size():
    image = Image.new("RGB", (512, 512), color=(0, 255, 0))

    canvas, transform = fit_image_to_square_canvas(image, 512, fill=(0, 0, 0))
    restored = restore_from_square_canvas(canvas, transform)

    assert canvas.size == (512, 512)
    assert transform.content_box == (0, 0, 512, 512)
    assert restored.size == image.size


def test_landscape_mask_uses_same_transform_as_image():
    image = Image.new("RGB", (1200, 900), color=(0, 0, 255))
    mask = Image.new("L", image.size, color=255)

    canvas, transform = fit_image_to_square_canvas(image, 512, fill=(0, 0, 0))
    mask_canvas = apply_square_canvas_transform(mask, transform, fill=0)
    restored = restore_from_square_canvas(canvas, transform)

    assert transform.resized_size == (512, 384)
    assert mask_canvas.size == (512, 512)
    assert restored.size == image.size
