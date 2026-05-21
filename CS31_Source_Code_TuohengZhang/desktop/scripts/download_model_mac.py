#!/usr/bin/env python3
"""Standalone script to download the SD 1.5 Inpainting base model.

Run this from Terminal if the in-app download fails:

    python3 download_model_mac.py

The model (~4 GB) is saved to:
    ~/Library/Application Support/CS31-1-Rhinoplasty-Prediction-Studio/

Once complete, relaunch the app — it will skip the download step.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path


REPO_ID = "botp/stable-diffusion-v1-5-inpainting"
TARGET_DIR = (
    Path.home()
    / "Library"
    / "Application Support"
    / "CS31-1-Rhinoplasty-Prediction-Studio"
)
ALLOW_PATTERNS = [
    "model_index.json",
    "unet/*.bin",
    "unet/config.json",
    "vae/*.bin",
    "vae/config.json",
    "text_encoder/*.bin",
    "text_encoder/*.json",
    "tokenizer/*",
    "scheduler/*",
    "feature_extractor/*",
]


def check_deps() -> bool:
    missing = []
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing.append("tqdm")
    if missing:
        print("Missing packages. Install them first:")
        print(f"  pip3 install {' '.join(missing)}")
        return False
    return True


def already_downloaded(target: Path) -> bool:
    unet = target / "unet" / "config.json"
    vae = target / "vae" / "config.json"
    return unet.exists() and vae.exists()


def main() -> int:
    print("CS31 — SD 1.5 Inpainting model downloader")
    print(f"Destination: {TARGET_DIR}")
    print()

    if not check_deps():
        return 1

    if already_downloaded(TARGET_DIR):
        print("Model already downloaded. Nothing to do.")
        print("If the app still shows the download dialog, relaunch it.")
        return 0

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    from huggingface_hub import snapshot_download

    print("Downloading ~4 GB from huggingface.co ...")
    print("This may take 10–30 minutes depending on your connection.")
    print()

    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(TARGET_DIR),
            allow_patterns=ALLOW_PATTERNS,
        )
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run the script again to resume.")
        return 1
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("Check your internet connection and try again.")
        return 1

    print()
    print("Download complete.")
    print("Relaunch CS31-1-Rhinoplasty-Prediction-Studio — the app will open directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
