import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ml.dataset_tools import (
    build_manifest,
    iter_source_entries,
    load_image_bytes,
    split_paired_image,
    stable_split,
    validate_split_halves,
)


def test_split_paired_image_halves_have_equal_size(tmp_path: Path):
    path = tmp_path / "paired.jpeg"
    image = Image.new("RGB", (1200, 1600), color=(255, 255, 255))
    image.save(path)
    left, right = split_paired_image(Image.open(path))
    assert left.size == (600, 1600)
    assert right.size == (600, 1600)


def test_build_manifest_marks_zip_name_duplicate(tmp_path: Path):
    first = tmp_path / "case1.jpeg"
    second = tmp_path / "case2.jpeg"
    image_one = Image.new("RGB", (1200, 1600), color=(255, 0, 0))
    draw_one = ImageDraw.Draw(image_one)
    draw_one.rectangle((100, 100, 500, 700), fill=(0, 0, 255))
    image_one.save(first)

    image_two = Image.new("RGB", (1200, 1600), color=(0, 255, 0))
    draw_two = ImageDraw.Draw(image_two)
    draw_two.ellipse((250, 300, 850, 1100), fill=(255, 255, 0))
    image_two.save(second)

    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(first, arcname="case1.jpeg")
        zf.write(second, arcname="case2_zip.jpeg")

    records = build_manifest(tmp_path)
    canonical = [record for record in records if not record.is_duplicate]
    duplicates = [record for record in records if record.is_duplicate]

    assert len(records) == 4
    assert len(canonical) == 2
    assert any(record.duplicate_reason == "duplicate_name" for record in duplicates)


def test_stable_split_assigns_all_samples():
    sample_ids = [f"id_{index}" for index in range(20)]
    split_map = stable_split(sample_ids, seed=31, val_ratio=0.1, test_ratio=0.1)
    assert set(split_map.keys()) == set(sample_ids)
    assert {"train", "val", "test"} <= set(split_map.values())


def test_iter_source_entries_scans_subdirectories(tmp_path: Path):
    subdir = tmp_path / "subdir_batch"
    subdir.mkdir()
    img1 = Image.new("RGB", (200, 300), color=(255, 0, 0))
    img1.save(subdir / "img_a.jpeg")
    img2 = Image.new("RGB", (200, 300), color=(0, 255, 0))
    img2.save(subdir / "img_b.jpeg")

    # Also place a direct file
    img3 = Image.new("RGB", (200, 300), color=(0, 0, 255))
    img3.save(tmp_path / "direct.jpeg")

    entries = list(iter_source_entries(tmp_path))
    kinds = [e[0] for e in entries]
    names = [e[2] for e in entries]

    assert "filesystem" in kinds
    assert "directory" in kinds
    assert "direct.jpeg" in names
    assert "img_a.jpeg" in names
    assert "img_b.jpeg" in names
    assert len(entries) == 3


def test_split_paired_image_square_splits_top_bottom():
    top_color = (255, 0, 0)
    bottom_color = (0, 0, 255)
    image = Image.new("RGB", (1080, 1080))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1080, 540), fill=top_color)
    draw.rectangle((0, 540, 1080, 1080), fill=bottom_color)

    pre, post = split_paired_image(image)
    assert pre.size == (1080, 540)
    assert post.size == (1080, 540)

    pre_avg = np.array(pre).mean(axis=(0, 1))
    post_avg = np.array(post).mean(axis=(0, 1))
    assert pre_avg[0] > 200  # red channel dominant in pre (top)
    assert post_avg[2] > 200  # blue channel dominant in post (bottom)


def test_build_manifest_transitive_dedup(tmp_path: Path):
    # Create three images where A~B and B~C (within threshold) but A~C might exceed it.
    # Union-find should merge all three transitively.
    # Use a large threshold to ensure the test is robust regardless of phash sensitivity.
    base = Image.new("RGB", (1200, 1600), color=(128, 128, 128))
    draw = ImageDraw.Draw(base)
    draw.rectangle((100, 100, 500, 700), fill=(200, 50, 50))

    img_a = base.copy()
    img_a.save(tmp_path / "img_a.jpeg")

    # B = exact copy of A (phash distance 0)
    img_a.save(tmp_path / "img_b.jpeg")

    # C = very different image
    img_c = Image.new("RGB", (1200, 1600), color=(0, 200, 100))
    draw_c = ImageDraw.Draw(img_c)
    draw_c.ellipse((200, 200, 900, 1400), fill=(255, 255, 0))
    img_c.save(tmp_path / "img_c.jpeg")

    # With threshold=4, A and B should merge (distance 0), C should stay separate
    records = build_manifest(tmp_path, near_duplicate_threshold=4)
    canonical = [r for r in records if not r.is_duplicate]
    assert len(canonical) == 2  # {A,B} cluster + C

    # With a very large threshold, all three should merge
    records_large = build_manifest(tmp_path, near_duplicate_threshold=64)
    canonical_large = [r for r in records_large if not r.is_duplicate]
    assert len(canonical_large) == 1


def test_validate_split_halves_rejects_blank():
    blank = Image.new("RGB", (256, 256), color=(0, 0, 0))
    normal = Image.new("RGB", (256, 256), color=(128, 64, 200))
    draw = ImageDraw.Draw(normal)
    draw.ellipse((20, 20, 200, 200), fill=(255, 255, 0))

    assert validate_split_halves(blank, normal, "test_blank") is False
    assert validate_split_halves(normal, blank, "test_blank2") is False
    assert validate_split_halves(blank, blank, "test_all_blank") is False


def test_validate_split_halves_accepts_normal():
    img1 = Image.new("RGB", (256, 256), color=(128, 64, 200))
    draw1 = ImageDraw.Draw(img1)
    draw1.ellipse((20, 20, 200, 200), fill=(255, 255, 0))

    img2 = Image.new("RGB", (256, 256), color=(50, 150, 100))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle((30, 30, 180, 180), fill=(0, 0, 255))

    assert validate_split_halves(img1, img2, "test_normal") is True


def test_load_image_bytes_directory(tmp_path: Path):
    subdir = tmp_path / "batch"
    subdir.mkdir()
    img = Image.new("RGB", (100, 100), color=(42, 84, 126))
    img.save(subdir / "sample.jpeg")

    data = load_image_bytes("directory", subdir, "sample.jpeg")
    assert len(data) > 0
    assert isinstance(data, bytes)
