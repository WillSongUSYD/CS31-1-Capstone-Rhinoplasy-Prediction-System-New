# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CS31-1-Rhinoplasty-Prediction-Studio on Windows.

Build from CS31_Source_Code_TuohengZhang/:

    pyinstaller desktop/CS31_windows.spec

Produces dist/CS31-1-Rhinoplasty-Prediction-Studio/ directory.
Zip it for distribution (7-Zip or Windows built-in).

First-launch behaviour is the same as macOS: the app shows an onboarding
dialog and downloads the ~4 GB SD 1.5 Inpainting base model from
huggingface.co. The V6 LoRA (25 MB) is bundled inside the distribution.
"""
# noqa: F821 -- Analysis, PYZ, EXE, COLLECT are injected by PyInstaller
from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

SPECPATH = Path(SPECPATH)  # type: ignore[name-defined]  # noqa: F821
HERE = SPECPATH          # desktop/
REPO = HERE.parent       # CS31_Source_Code_TuohengZhang/

APP_NAME = "CS31-1-Rhinoplasty-Prediction-Studio"

# Collect mediapipe data files (tflite models, etc.) and hidden imports.
mp_datas, mp_bins, mp_hiddens = collect_all("mediapipe")

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
    binaries=mp_bins,
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
    ] + mp_datas + insightface_datas,
    hiddenimports=[
        # PyTorch lazy submodules.
        "torch._C", "torch.utils", "torch.nn", "torch.nn.functional",
        "torch.backends.mps",
        # Hugging Face ecosystem.
        "diffusers", "transformers", "peft", "safetensors", "accelerate",
        "huggingface_hub", "huggingface_hub.file_download",
        "huggingface_hub.snapshot_download", "tokenizers",
        # Networking.
        "requests", "urllib3", "certifi", "charset_normalizer", "idna",
        # File / packaging helpers.
        "filelock", "packaging", "regex", "tqdm", "fsspec", "yaml",
        "typing_extensions", "importlib_metadata",
        # ML deps.
        "sklearn", "sklearn.utils", "sklearn.neighbors", "cv2", "onnxruntime",
        # Our packages.
        "backend", "backend.inference", "backend.inference_sd",
        "backend.db",
        "ml", "ml.config", "ml.nose_roi", "ml.landmarks",
        "ml.description", "ml.runtime",
        "desktop", "desktop.core", "desktop.core.config",
        "desktop.core.paths", "desktop.core.downloader",
        "desktop.core.inference_worker", "desktop.core.image_geometry",
        "desktop.core.validator", "desktop.core.device_probe",
        "desktop.widgets", "desktop.widgets.before_after",
        "desktop.widgets.busy_overlay", "desktop.widgets.drop_zone",
        "desktop.widgets.onboarding_dialog",
        "desktop.widgets.validation_report",
        "ml.models.sd_inpaint",
        # Qt.
        "PyQt6.QtCore", "PyQt6.QtGui", "PyQt6.QtWidgets",
        "PyQt6.QtNetwork",
        # Standard library items py2app includes explicitly.
        "sqlite3", "queue", "concurrent.futures",
    ] + mp_hiddens,
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
