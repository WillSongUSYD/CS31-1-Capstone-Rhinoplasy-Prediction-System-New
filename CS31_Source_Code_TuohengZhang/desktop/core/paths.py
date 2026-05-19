"""Path resolution for bundled vs user-writable locations.

macOS app bundles are READ-ONLY once signed/installed. Any file the app
needs to write (downloaded SD base model, generated result images, config
cache) must go to ``~/Library/Application Support/CS31Preview/``.

This module exposes two families:

* ``bundle_*()`` — resources shipped inside the ``.app`` (read-only):
  V6 LoRA, face_landmarker.task, bundled InsightFace buffalo_l.

* ``user_*()`` — user-writable state under Application Support:
  downloaded SD base, config.json, saved outputs, logs.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


_APP_NAME = "CS31Preview"


def _is_frozen() -> bool:
    """Return True when running inside a py2app/PyInstaller bundle.

    py2app sets ``sys.frozen`` to the string ``"macosx_app"``. In that case
    ``__file__`` lives inside ``.../Contents/Resources/lib/python3.X/``,
    so resource-relative lookups must be rooted at the bundle's
    ``Resources`` directory rather than the source tree.
    """
    return hasattr(sys, "frozen") and sys.frozen


def bundle_root() -> Path:
    """Root of bundled read-only resources.

    Source-tree layout: ``<repo>/desktop/``
    Frozen bundle: ``<App>.app/Contents/Resources/desktop/``
    """
    if _is_frozen():
        # py2app copies the entire ``desktop`` package into Resources.
        # sys.executable → <App>.app/Contents/MacOS/CS31Preview
        app_root = Path(sys.executable).resolve().parent.parent
        return app_root / "Resources" / "desktop"
    return Path(__file__).resolve().parents[1]


def bundle_lora() -> Path:
    """V6 LoRA safetensors shipped inside the .app."""
    return bundle_root() / "bundled_models" / "lora"


def bundle_insightface() -> Path:
    """InsightFace buffalo_l shipped inside the .app.

    We set ``INSIGHTFACE_HOME`` to this path at startup so the library
    finds its ONNX models without trying to download them (first-run
    no-network case).
    """
    return bundle_root() / "bundled_models" / "insightface"


def bundle_face_landmarker() -> Path:
    """MediaPipe face_landmarker.task file."""
    return bundle_root() / "bundled_models" / "face_landmarker.task"


def user_support_dir() -> Path:
    """Platform-appropriate writable app-data directory. Created on first access.

    - Windows: ``%APPDATA%\\CS31Preview\\``
    - macOS:   ``~/Library/Application Support/CS31Preview/``
    """
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        d = Path(appdata) / _APP_NAME if appdata else Path.home() / _APP_NAME
    else:
        d = Path.home() / "Library" / "Application Support" / _APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_sd_base_dir() -> Path:
    """Where the 4GB SD 1.5 Inpainting base model lives after first-run
    download."""
    d = user_support_dir() / "models" / "sd_base" / "inpaint"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_predictions_dir() -> Path:
    """Scratch area for per-request intermediate files. Most of the time
    we keep everything in memory (see ``InferenceWorker``), but a few
    existing modules inside ``backend.inference`` still write to disk;
    we point them here via ``CS31_PREDICTIONS_DIR``.
    """
    d = user_support_dir() / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_config_path() -> Path:
    """Path to ``config.json`` (e.g. ``force_cpu`` decision after MPS
    canary)."""
    return user_support_dir() / "config.json"


def user_output_dir() -> Path:
    """Where the Save button writes generated post-op images.

    ``~/Pictures/CS31Preview/`` is a more discoverable default than the
    app's Application Support dir for user-facing outputs.
    """
    d = Path.home() / "Pictures" / _APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_sd_base_present() -> bool:
    """Quick existence check used at launch to decide whether to show
    the onboarding download dialog. Checks for the UNet weights file
    specifically since it's the largest and downloaded last - if it's
    there, the rest of the repo is almost certainly complete.
    """
    unet_dir = user_sd_base_dir() / "unet"
    # Accept either .safetensors or .bin format (the botp mirror ships .bin).
    return (unet_dir / "diffusion_pytorch_model.safetensors").exists() \
        or (unet_dir / "diffusion_pytorch_model.bin").exists()
