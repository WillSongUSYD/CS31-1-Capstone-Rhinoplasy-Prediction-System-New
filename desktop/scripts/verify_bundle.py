"""Post-build sanity checks for the produced .app bundle.

Run as ``python desktop/scripts/verify_bundle.py dist/CS31Preview.app``.
Fails with a non-zero exit code if any required file is missing or
looks suspicious (e.g. LoRA file is a few bytes long — probably a
symlink py2app didn't follow).

This saves users from shipping a broken .app to QA or end-users.
"""
from __future__ import annotations

import plistlib
import sys
from pathlib import Path


REQUIRED_FILES = [
    # (relative path inside .app, min size in bytes, description)
    ("Contents/Info.plist", 500, "Info.plist"),
    ("Contents/MacOS/CS31Preview", 1024, "launcher executable"),
    (
        "Contents/Resources/desktop/bundled_models/lora/pytorch_lora_weights.safetensors",
        20 * 1024 * 1024,  # ≥20 MB (V6 LoRA is ~25 MB)
        "V6 LoRA weights",
    ),
    (
        "Contents/Resources/desktop/assets/style.qss",
        200,
        "QSS stylesheet",
    ),
]


def main(app_path: str) -> int:
    app = Path(app_path)
    if not app.is_dir():
        print(f"ERROR: {app} is not a directory", file=sys.stderr)
        return 2

    errors: list[str] = []

    for rel, min_size, desc in REQUIRED_FILES:
        p = app / rel
        if not p.exists():
            errors.append(f"MISSING: {desc} ({rel})")
            continue
        size = p.stat().st_size
        if size < min_size:
            errors.append(
                f"TOO SMALL: {desc} is {size} B < expected {min_size} B"
                f" — likely a broken symlink"
            )

    # Check Info.plist for minimum macOS version.
    info_plist = app / "Contents" / "Info.plist"
    if info_plist.exists():
        try:
            data = plistlib.loads(info_plist.read_bytes())
            min_sys = data.get("LSMinimumSystemVersion", "")
            if not min_sys:
                errors.append("Info.plist missing LSMinimumSystemVersion")
            bundle_id = data.get("CFBundleIdentifier", "")
            if bundle_id != "com.cs31.preview":
                errors.append(f"Info.plist CFBundleIdentifier = {bundle_id!r}")
        except Exception as exc:
            errors.append(f"Info.plist unreadable: {exc}")

    # Verify torch was bundled inside (not linked out to brew Python).
    torch_dir = list(
        (app / "Contents" / "Resources" / "lib").glob("python*/torch")
    )
    if not torch_dir:
        errors.append(
            "torch/ not found in Contents/Resources/lib/python*/ — py2app may"
            " have failed to collect it; the .app will crash at launch when"
            " it imports backend.inference_sd"
        )

    if errors:
        print("VERIFY FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"verify OK: {app}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: verify_bundle.py <path/to/App.app>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
