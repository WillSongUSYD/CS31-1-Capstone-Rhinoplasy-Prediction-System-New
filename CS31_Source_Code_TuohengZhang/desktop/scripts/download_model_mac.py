#!/usr/bin/env python3
"""Standalone script to download the SD 1.5 Inpainting base model.

Run this from Terminal if the in-app download fails:

    python3 download_model_mac.py

No prior setup needed — the script installs its own dependencies.
The model (~4 GB) is saved to:
    ~/Library/Application Support/CS31-1-Rhinoplasty-Prediction-Studio/models/sd_base/inpaint/

Once complete, relaunch the app — it will skip the download step.
"""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


REPO_ID = "botp/stable-diffusion-v1-5-inpainting"
TARGET_DIR = (
    Path.home()
    / "Library"
    / "Application Support"
    / "CS31-1-Rhinoplasty-Prediction-Studio"
    / "models"
    / "sd_base"
    / "inpaint"
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


def ensure_deps() -> None:
    """Install huggingface_hub and tqdm if not already available.

    Tries three strategies in order:
      1. Normal pip install (works if already in a venv or user site)
      2. pip install --user (avoids system-wide write, works on most setups)
      3. pip install --break-system-packages (Homebrew Python on macOS 13+)
    """
    needed = []
    for pkg in ("huggingface_hub", "tqdm"):
        try:
            __import__(pkg)
        except ImportError:
            needed.append(pkg)

    if not needed:
        return

    print(f"Installing required packages: {', '.join(needed)}")

    strategies = [
        [sys.executable, "-m", "pip", "install", "--quiet"] + needed,
        [sys.executable, "-m", "pip", "install", "--quiet", "--user"] + needed,
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--break-system-packages"] + needed,
    ]

    for cmd in strategies:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            # Re-add user site-packages to path so the fresh install is visible.
            import site
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.insert(0, user_site)
            print("Packages installed.\n")
            return

    print("ERROR: Could not install required packages automatically.")
    print("Please run one of the following manually, then re-run this script:")
    print(f"  pip3 install --user {' '.join(needed)}")
    print(f"  pip3 install --break-system-packages {' '.join(needed)}")
    sys.exit(1)


def already_downloaded(target: Path) -> bool:
    unet_dir = target / "unet"
    return (
        (unet_dir / "diffusion_pytorch_model.safetensors").exists()
        or (unet_dir / "diffusion_pytorch_model.bin").exists()
    )


def main() -> int:
    print("CS31 — SD 1.5 Inpainting model downloader")
    print(f"Destination: {TARGET_DIR}")
    print()

    ensure_deps()

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
