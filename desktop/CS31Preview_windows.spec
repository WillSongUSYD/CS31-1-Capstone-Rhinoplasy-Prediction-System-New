# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building CS31Preview on Windows.

Run on a Windows machine from the project root:

    python -m PyInstaller desktop\CS31Preview_windows.spec --noconfirm --clean

The Mac build still uses desktop/setup.py + py2app. This spec is only for
Windows and intentionally does not replace the existing Mac app build.
"""
from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


ROOT = Path(SPECPATH).resolve().parent.parent


datas = [
    (
        str(ROOT / "desktop" / "assets" / "style.qss"),
        "desktop/assets",
    ),
    (
        str(
            ROOT
            / "models"
            / "outcome_v3_512"
            / "sd_inpaint_nose_v6"
            / "step_10000"
            / "pytorch_lora_weights.safetensors"
        ),
        "desktop/bundled_models/lora",
    ),
]
binaries = []
hiddenimports = ["sqlite3", "queue"]


def collect_package(package_name: str) -> None:
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)


for package in [
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
    collect_package(package)


insightface_dir = ROOT / "desktop" / "bundled_models" / "insightface"
if insightface_dir.exists():
    datas.append((str(insightface_dir), "desktop/bundled_models/insightface"))


a = Analysis(
    [str(ROOT / "desktop" / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib.tests",
        "pytest",
        "tkinter",
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtWebEngineWidgets",
        "backend.serve",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CS31Preview",
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="CS31Preview",
)
