"""Post-build sanity checks for the Windows PyInstaller dist folder.

Run as:
    python desktop/scripts/verify_bundle_windows.py dist/CS31-1-Rhinoplasty-Prediction-Studio

Exits non-zero if any required file is missing or suspiciously small.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "CS31-1-Rhinoplasty-Prediction-Studio"

REQUIRED_FILES = [
    # (relative path inside dist folder, min size in bytes, description)
    (f"{APP_NAME}.exe", 1024, "launcher executable"),
    (
        f"desktop/bundled_models/lora/pytorch_lora_weights.safetensors",
        20 * 1024 * 1024,
        "V6 LoRA weights",
    ),
    (
        "desktop/assets/style.qss",
        200,
        "QSS stylesheet",
    ),
    (
        "download_sd_model_v3.bat",
        100,
        "fallback model downloader",
    ),
]


def main(dist_path: str) -> int:
    dist = Path(dist_path)
    if not dist.is_dir():
        print(f"ERROR: {dist} is not a directory", file=sys.stderr)
        return 2

    errors: list[str] = []

    for rel, min_size, desc in REQUIRED_FILES:
        p = dist / rel
        if not p.exists():
            errors.append(f"MISSING: {desc} ({rel})")
            continue
        size = p.stat().st_size
        if size < min_size:
            errors.append(
                f"TOO SMALL: {desc} is {size} B < expected {min_size} B"
                f" — likely incomplete bundle"
            )

    # Check torch was collected (without it SD inference crashes at import).
    torch_dirs = list(dist.rglob("torch/nn/__init__.py"))
    if not torch_dirs:
        errors.append(
            "torch not found inside dist folder — PyInstaller may have "
            "failed to collect it; the exe will crash on launch"
        )

    if errors:
        print("VERIFY FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"verify OK: {dist}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: verify_bundle_windows.py <path/to/{APP_NAME}>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
