import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .config import MANIFEST_PATH, SPLITS_PATH

logger = logging.getLogger(__name__)


@dataclass
class PairItem:
    sample_id: str
    pre_path: str
    post_path: str
    split: str


def load_pairs(split: str, limit: Optional[int] = None, exclude_frontal: bool = True) -> List[PairItem]:
    manifest = pd.read_csv(MANIFEST_PATH)
    splits = pd.read_csv(SPLITS_PATH)
    usable = manifest[(manifest["is_duplicate"] == False) & manifest["pre_path"].notna() & manifest["post_path"].notna()]  # noqa: E712

    if exclude_frontal and "view_type" in manifest.columns:
        usable = usable[usable["view_type"] != "frontal"]

    merged = usable.merge(splits, on="sample_id", how="inner")
    rows = merged[merged["split"] == split].copy()
    if limit:
        rows = rows.head(limit)
    return [
        PairItem(sample_id=row["sample_id"], pre_path=row["pre_path"], post_path=row["post_path"], split=row["split"])
        for _, row in rows.iterrows()
    ]


class PairImageDataset(Dataset):
    def __init__(self, split: str, limit: Optional[int] = None, image_size: int = 256, augment: bool = False):
        self.items = load_pairs(split=split, limit=limit)
        # Skip any manifest entries whose referenced files are missing on
        # disk (e.g. partial dataset rsync). Fail quietly rather than at
        # __getitem__ so DataLoader workers don't raise mid-epoch.
        total_before = len(self.items)
        self.items = [
            item for item in self.items
            if Path(item.pre_path).exists() and Path(item.post_path).exists()
        ]
        dropped = total_before - len(self.items)
        if dropped > 0:
            logger.warning(
                "Filtered %d of %d %s items (split=%s) due to missing files",
                dropped, total_before, self.__class__.__name__, split,
            )
        if len(self.items) == 0:
            raise RuntimeError(
                f"{self.__class__.__name__} for split={split!r} is empty after filtering. "
                f"Check that the prepared pairs exist on disk (run ml.prepare_pairs)."
            )
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        pre_image = Image.open(Path(item.pre_path)).convert("RGB")
        post_image = Image.open(Path(item.post_path)).convert("RGB")

        if self.augment:
            # Synchronized horizontal flip
            if random.random() > 0.5:
                pre_image = TF.hflip(pre_image)
                post_image = TF.hflip(post_image)
            # Color jitter (same random params for both)
            fn_idx, brightness, contrast, saturation, hue = transforms.ColorJitter.get_params(
                self.color_jitter.brightness, self.color_jitter.contrast,
                self.color_jitter.saturation, self.color_jitter.hue,
            )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness is not None:
                    pre_image = TF.adjust_brightness(pre_image, brightness)
                    post_image = TF.adjust_brightness(post_image, brightness)
                elif fn_id == 1 and contrast is not None:
                    pre_image = TF.adjust_contrast(pre_image, contrast)
                    post_image = TF.adjust_contrast(post_image, contrast)
                elif fn_id == 2 and saturation is not None:
                    pre_image = TF.adjust_saturation(pre_image, saturation)
                    post_image = TF.adjust_saturation(post_image, saturation)
                elif fn_id == 3 and hue is not None:
                    pre_image = TF.adjust_hue(pre_image, hue)
                    post_image = TF.adjust_hue(post_image, hue)

        return {
            "sample_id": item.sample_id,
            "pre": self.base_transform(pre_image),
            "post": self.base_transform(post_image),
        }


class NoseROIDataset(Dataset):
    """Dataset that loads pre-extracted nose ROI crops.

    REQUIRES a source directory that matches the requested ``image_size``
    exactly (e.g. ``nose_roi_128``). Falling back to a different resolution
    silently would mean either throwing away pixels (downsample) or
    fabricating detail (upsample) - both have contributed to subtle
    training bugs. Instead, we fail fast with a pointer to the CLI tool.
    """

    def __init__(self, split: str, limit: Optional[int] = None, image_size: int = 128, augment: bool = False):
        from .config import ARTIFACTS_DIR
        self.roi_dir = ARTIFACTS_DIR / "dataset" / f"nose_roi_{image_size}"
        if not self.roi_dir.exists() or not any(self.roi_dir.iterdir()):
            raise FileNotFoundError(
                f"Nose ROI directory {self.roi_dir} is missing or empty. "
                f"Generate it with: python -m ml.nose_roi --size {image_size}"
            )
        self.items = load_pairs(split=split, limit=limit)
        # Filter to items that have BOTH pre and post nose ROI files on disk
        # (see #12: skip incomplete pairs rather than crashing at __getitem__).
        total_before = len(self.items)
        self.items = [
            item for item in self.items
            if (self.roi_dir / f"{item.sample_id}_pre.jpg").exists()
            and (self.roi_dir / f"{item.sample_id}_post.jpg").exists()
        ]
        dropped = total_before - len(self.items)
        if dropped > 0:
            logger.warning(
                "Filtered %d of %d %s items (split=%s) due to missing files",
                dropped, total_before, self.__class__.__name__, split,
            )
        if len(self.items) == 0:
            raise RuntimeError(
                f"{self.__class__.__name__} for split={split!r} is empty after filtering. "
                f"Re-generate nose ROI crops with: python -m ml.nose_roi --size {image_size}"
            )
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        pre_image = Image.open(self.roi_dir / f"{item.sample_id}_pre.jpg").convert("RGB")
        post_image = Image.open(self.roi_dir / f"{item.sample_id}_post.jpg").convert("RGB")

        if self.augment:
            if random.random() > 0.5:
                pre_image = TF.hflip(pre_image)
                post_image = TF.hflip(post_image)
            fn_idx, brightness, contrast, saturation, hue = transforms.ColorJitter.get_params(
                self.color_jitter.brightness, self.color_jitter.contrast,
                self.color_jitter.saturation, self.color_jitter.hue,
            )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness is not None:
                    pre_image = TF.adjust_brightness(pre_image, brightness)
                    post_image = TF.adjust_brightness(post_image, brightness)
                elif fn_id == 1 and contrast is not None:
                    pre_image = TF.adjust_contrast(pre_image, contrast)
                    post_image = TF.adjust_contrast(post_image, contrast)
                elif fn_id == 2 and saturation is not None:
                    pre_image = TF.adjust_saturation(pre_image, saturation)
                    post_image = TF.adjust_saturation(post_image, saturation)
                elif fn_id == 3 and hue is not None:
                    pre_image = TF.adjust_hue(pre_image, hue)
                    post_image = TF.adjust_hue(post_image, hue)

        return {
            "sample_id": item.sample_id,
            "pre": self.base_transform(pre_image),
            "post": self.base_transform(post_image),
        }


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu() * 0.5 + 0.5
