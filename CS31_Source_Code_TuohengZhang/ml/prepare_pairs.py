import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .alignment import align_face, generate_nose_mask
from .config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
    MANIFEST_PATH,
    MASK_DIR,
    PAIR_256_DIR,
    PAIR_ALIGNED_DIR,
    PAIR_FULL_DIR,
    SPLITS_PATH,
    ensure_directories,
)
from .dataset_tools import load_image, split_paired_image, stable_split, validate_split_halves
from .landmarks import detect_view_type

logger = logging.getLogger(__name__)


def resize_keep_aspect(image: Image.Image, target_size: int) -> Image.Image:
    """Resize image to fit within target_size x target_size, keeping aspect ratio, pad with black."""
    w, h = image.size
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def prepare_pairs(image_size: int = DEFAULT_IMAGE_SIZE) -> None:
    ensure_directories()
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}. Run python -m ml.index_dataset first.")

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["pre_path"] = manifest["pre_path"].fillna("").astype(str)
    manifest["post_path"] = manifest["post_path"].fillna("").astype(str)
    if "view_type" not in manifest.columns:
        manifest["view_type"] = ""
    canonical = manifest[manifest["is_duplicate"] == False].copy()  # noqa: E712

    valid_sample_ids = []
    pre_sample_ids = []
    view_types = []
    view_counts = {"profile": 0, "frontal": 0, "unknown": 0}

    for row in tqdm(canonical.to_dict("records"), desc="Preparing pairs"):
        image = load_image(row["source_kind"], Path(row["source_container"]), row.get("source_member", "") or "")
        pre_image, post_image = split_paired_image(image)

        if not validate_split_halves(pre_image, post_image, row["sample_id"]):
            logger.warning("Skipping sample %s due to failed validation", row["sample_id"])
            continue

        valid_sample_ids.append(row["sample_id"])
        sid = row["sample_id"]

        # Save full resolution
        pre_image.save(PAIR_FULL_DIR / f"{sid}_pre.jpg", quality=95)
        post_image.save(PAIR_FULL_DIR / f"{sid}_post.jpg", quality=95)

        # Save aspect-preserved resize (for backward compatibility)
        pre_resized = resize_keep_aspect(pre_image, image_size)
        pre_resized.save(PAIR_256_DIR / f"{sid}_pre.jpg", quality=95)
        resize_keep_aspect(post_image, image_size).save(PAIR_256_DIR / f"{sid}_post.jpg", quality=95)

        # Save aligned images (face-detected, cropped, centered)
        pre_aligned = align_face(pre_image, image_size)
        post_aligned = align_face(post_image, image_size)
        pre_aligned.save(PAIR_ALIGNED_DIR / f"{sid}_pre.jpg", quality=95)
        post_aligned.save(PAIR_ALIGNED_DIR / f"{sid}_post.jpg", quality=95)

        # Generate and save nose mask (from aligned pre image)
        mask = generate_nose_mask(pre_aligned, image_size)
        mask_uint8 = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(MASK_DIR / f"{sid}_mask.png")

        manifest.loc[manifest["sample_id"] == sid, "pre_path"] = str(PAIR_ALIGNED_DIR / f"{sid}_pre.jpg")
        manifest.loc[manifest["sample_id"] == sid, "post_path"] = str(PAIR_ALIGNED_DIR / f"{sid}_post.jpg")

        # Detect view type inline on the aligned image instead of accumulating
        # every PIL image in a list until the end of the loop - the old
        # approach held ~N * (image_size^2 * 3) bytes in RAM and would OOM
        # on >1k sample datasets at 512x512.
        vt = detect_view_type(pre_aligned)
        manifest.loc[manifest["sample_id"] == sid, "view_type"] = vt
        if vt not in view_counts:
            view_counts[vt] = 0
        view_counts[vt] += 1
        view_types.append(vt)
        pre_sample_ids.append(sid)

    print(f"View types: profile={view_counts['profile']}, frontal={view_counts['frontal']}, unknown={view_counts['unknown']}")

    manifest.to_csv(MANIFEST_PATH, index=False)

    # Only include non-frontal samples in splits
    trainable_ids = [
        sid for sid, vt in zip(pre_sample_ids, view_types) if vt != "frontal"
    ]
    print(f"Trainable samples (profile+unknown): {len(trainable_ids)}")

    split_map = stable_split(
        sample_ids=trainable_ids,
        seed=DEFAULT_SPLIT_SEED,
        val_ratio=DEFAULT_VAL_RATIO,
        test_ratio=DEFAULT_TEST_RATIO,
    )
    with SPLITS_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_id", "split"])
        writer.writeheader()
        for sample_id, split in split_map.items():
            writer.writerow({"sample_id": sample_id, "split": split})


def main() -> None:
    parser = argparse.ArgumentParser(description="Split paired rhinoplasty images into pre/post samples.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    args = parser.parse_args()
    prepare_pairs(image_size=args.image_size)
    print(f"Wrote aligned pairs to {PAIR_ALIGNED_DIR}")
    print(f"Wrote nose masks to {MASK_DIR}")
    print(f"Wrote split definitions to {SPLITS_PATH}")


if __name__ == "__main__":
    main()
