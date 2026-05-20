# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CS31-1-Rhinoplasty-Prediction-Studio on Windows.

Build from CS31_Source_Code_TuohengZhang/:

    pyinstaller desktop/CS31_windows.spec

Produces dist/CS31-1-Rhinoplasty-Prediction-Studio/ directory.

First-launch behaviour is the same as macOS: the app shows an onboarding
dialog and downloads the ~4 GB SD 1.5 Inpainting base model from
huggingface.co. The V6 LoRA and the InsightFace buffalo_l models are
bundled inside the distribution.

Collection strategy: third-party packages are pulled in with collect_all()
rather than relying on PyInstaller's static analysis. Packages with dynamic
imports or native DLLs — notably insightface (dynamic model_zoo imports)
and onnxruntime — are under-collected by static analysis, which silently
breaks face detection at runtime. This list mirrors the known-good
CS31Preview_windows.spec recipe.
"""
# noqa: F821 -- Analysis, PYZ, EXE, COLLECT are injected by PyInstaller
from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

SPECPATH = Path(SPECPATH)  # type: ignore[name-defined]  # noqa: F821
HERE = SPECPATH          # desktop/
REPO = HERE.parent       # CS31_Source_Code_TuohengZhang/

APP_NAME = "CS31-1-Rhinoplasty-Prediction-Studio"

# Fully collect third-party packages (submodules + data files + native
# binaries). Mirrors the known-good CS31Preview_windows.spec.
_pkg_datas = []
_pkg_binaries = []
_pkg_hiddenimports = []
for _pkg in [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "peft",
    "safetensors",
    "insightface",
    "onnxruntime",
    "cv2",
    "PIL",
    "numpy",
    "huggingface_hub",
    "tokenizers",
    "sklearn",
    "requests",
    "urllib3",
    "certifi",
    "charset_normalizer",
    "idna",
    "filelock",
    "packaging",
    "yaml",
    "regex",
    "tqdm",
    "fsspec",
    "typing_extensions",
    "PyQt6",
]:
    _d, _b, _h = collect_all(_pkg)
    _pkg_datas += _d
    _pkg_binaries += _b
    _pkg_hiddenimports += _h

# InsightFace buffalo_l models — REQUIRED for face detection. The full
# pack is bundled to match the known-good distribution. The CI workflow
# and build_windows.bat download buffalo_l before this build runs.
_insightface_local = Path.home() / ".insightface" / "models" / "buffalo_l"
insightface_datas = []
if _insightface_local.is_dir() and any(_insightface_local.glob("*.onnx")):
    insightface_datas = [
        (str(p), "desktop/bundled_models/insightface/models/buffalo_l")
        for p in _insightface_local.glob("*.onnx")
    ]
else:
    print(
        "[spec] warning: buffalo_l ONNX models not found — the built app "
        "will FAIL face detection. Run the InsightFace download step first.",
        file=sys.stderr,
    )

a = Analysis(
    [str(HERE / "app.py")],
    pathex=[str(REPO)],
    binaries=_pkg_binaries,
    datas=[
        # V6 LoRA (required for inference).
        (
            str(
                REPO
                / "models"
                / "outcome_v3_512"
                / "sd_inpaint_nose_v6"
                / "step_10000"
                / "pytorch_lora_weights.safetensors"
            ),
            "desktop/bundled_models/lora",
        ),
        # QSS stylesheet.
        (str(HERE / "assets" / "style.qss"), "desktop/assets"),
    ] + insightface_datas + _pkg_datas,
    hiddenimports=[
        # The app's own packages. PyInstaller picks most up from app.py,
        # but list the deferred / string-imported ones explicitly.
        "backend", "backend.inference", "backend.inference_sd", "backend.db",
        "ml", "ml.config", "ml.nose_roi", "ml.landmarks",
        "ml.description", "ml.runtime", "ml.models.sd_inpaint",
        "desktop", "desktop.core", "desktop.core.config",
        "desktop.core.paths", "desktop.core.downloader",
        "desktop.core.inference_worker", "desktop.core.image_geometry",
        "desktop.core.validator", "desktop.core.device_probe",
        "desktop.widgets", "desktop.widgets.before_after",
        "desktop.widgets.busy_overlay", "desktop.widgets.drop_zone",
        "desktop.widgets.onboarding_dialog",
        "desktop.widgets.validation_report",
        # accelerate is imported lazily by diffusers.
        "accelerate",
        # Standard library items PyInstaller sometimes misses.
        "sqlite3", "queue", "concurrent.futures",
    ] + _pkg_hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib.tests",
        "pytest",
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtWebEngineWidgets",
        "backend.serve",  # FastAPI server not needed in desktop app.
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)  # type: ignore[name-defined]  # noqa: F821

exe = EXE(  # type: ignore[name-defined]  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="desktop/assets/icon.ico",  # Add a .ico file to assets/ to enable.
    # Keep all support files (DLLs, Python runtime, bundled data) inside an
    # _internal/ folder so the distribution's top level stays clean: only the
    # exe plus the workflow-added README / FIRST_LAUNCH / downloader bat.
    contents_directory="_internal",
)

coll = COLLECT(  # type: ignore[name-defined]  # noqa: F821
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        # Torch DLLs use internal compression incompatible with UPX.
        "torch_cpu.dll",
        "torch_python.dll",
        "torch_cuda.dll",
        "libopenblas*",
        "mkl_*.dll",
    ],
    name=APP_NAME,
)
