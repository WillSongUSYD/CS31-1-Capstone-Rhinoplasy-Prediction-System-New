"""py2app configuration for CS31-1-Rhinoplasty-Prediction-Studio.

Build from the project root:

    cd /Applications/CS31
    python desktop/setup.py py2app --arch arm64

Produces ``dist/CS31-1-Rhinoplasty-Prediction-Studio.app``. Size:
  * ~400 MB python + torch shared libs + PyQt6
  * +25 MB bundled V6 LoRA
  * +4 MB MediaPipe face_landmarker
  * +300 MB InsightFace buffalo_l (ONNX models)
  ≈ 700-800 MB total. SD 1.5 base (4 GB) is downloaded on first launch,
  not bundled.

Recipe notes:

* ``torch``, ``diffusers``, ``insightface``, ``mediapipe`` — all have
  C-extensions or data files py2app needs to chase down. We list them
  in ``packages`` so py2app collects the whole tree rather than
  individual ``includes``.
* ``PyQt6.QtWebEngine*`` is explicitly excluded — QtWebEngine is huge
  (~200 MB of Chromium) and we never use it.
* ``backend.serve`` is excluded to drop the FastAPI server code path
  (runs only via ``python -m backend.serve`` which the desktop app
  never invokes).

This file is intentionally self-contained: running it from *inside*
``desktop/`` would break relative paths. Always invoke from the repo
root so ``data_files`` pathnames resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup

# py2app uses modulegraph to walk the import tree. Deeply nested
# conditional imports in torch / transformers / diffusers blow past
# the default recursion limit of 1000. Bump it BEFORE setup() runs.
# 10000 is comfortably beyond real-world needs for our stack.
sys.setrecursionlimit(10000)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

APP_ENTRY = [str(HERE / "app.py")]


def _collect_dist_info() -> list[tuple[str, list[str]]]:
    """py2app strips .dist-info/METADATA by default. Transformers (and
    several other libs) use ``importlib.metadata.version(...)`` to
    validate dep versions at import time — without the metadata those
    calls raise ``PackageNotFoundError`` and the first
    ``from transformers import AutoImageProcessor`` aborts with a
    misleading cascade.

    Fix: copy every ``*.dist-info`` directory from our venv into the
    bundle's site-packages. ~500 KB total; cheap insurance against
    future lib tightening.
    """
    _pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_sp = Path(sys.executable).parent.parent / "lib" / _pyver / "site-packages"
    if not venv_sp.exists():
        # Fallback for cases where sys.executable is the driving python
        # rather than venv (e.g. CI). Best-effort.
        import site
        candidates = [Path(p) for p in site.getsitepackages()]
        candidates = [c for c in candidates if c.exists()]
        if not candidates:
            return []
        venv_sp = candidates[0]

    pairs: list[tuple[str, list[str]]] = []
    for dinfo in venv_sp.glob("*.dist-info"):
        # Each dist-info is its own destination subdir under the bundle's site-packages.
        dest_rel = f"lib/{_pyver}/{dinfo.name}"
        files = [str(p) for p in dinfo.iterdir() if p.is_file()]
        if files:
            pairs.append((dest_rel, files))
    return pairs

# -- Data files to embed inside Contents/Resources ---------------------
# Each tuple is (relative_dest_dir_inside_Resources, [source_paths]).
# py2app copies these verbatim. We reference them from the running app
# via ``desktop.core.paths.bundle_root()``.

DATA_FILES = [
    # Root of our bundled_models/ inside Resources/desktop/bundled_models/.
    (
        "desktop/bundled_models/lora",
        [
            str(
                REPO
                / "models"
                / "outcome_v3_512"
                / "sd_inpaint_nose_v6"
                / "step_10000"
                / "pytorch_lora_weights.safetensors"
            ),
        ],
    ),
    # MediaPipe face landmarker (optional - only used if the app needs
    # it for future landmark features; harmless to ship empty on repos
    # that don't have it).
    # (
    #     "desktop/bundled_models",
    #     [str(REPO / "models" / "face_landmarker.task")],
    # ),
    # QSS stylesheet + icon (py2app doesn't automatically follow
    # non-Python siblings of the main script).
    (
        "desktop/assets",
        [str(HERE / "assets" / "style.qss")],
    ),
]

# InsightFace buffalo_l is ~300MB across ~6 ONNX files. Bundle the whole
# directory if the user has already downloaded it locally. If not
# present we skip — the first launch will download to ~/.insightface/
# as a fallback.
_INSIGHTFACE_LOCAL = Path.home() / ".insightface" / "models" / "buffalo_l"
if _INSIGHTFACE_LOCAL.is_dir():
    DATA_FILES.append((
        "desktop/bundled_models/insightface/models/buffalo_l",
        [str(p) for p in _INSIGHTFACE_LOCAL.glob("*.onnx")],
    ))
else:
    print(
        "[setup] warning: ~/.insightface/models/buffalo_l not found — "
        "InsightFace will download at first launch (requires network)",
        file=sys.stderr,
    )

# Tack on all dist-info directories so importlib.metadata works in-bundle.
DATA_FILES.extend(_collect_dist_info())


PY2APP_OPTIONS = {
    "argv_emulation": False,
    "iconfile": str(HERE / "assets" / "icon.icns") if (HERE / "assets" / "icon.icns").exists() else None,
    "packages": [
        "torch",
        "diffusers",
        "transformers",
        "peft",
        "safetensors",
        "insightface",
        "mediapipe",
        "PyQt6",
        "PIL",
        "cv2",
        "numpy",
        "onnxruntime",
        "huggingface_hub",
        # Transitive deps that py2app doesn't always pick up through
        # huggingface_hub's __getattr__ lazy loader. Without these the
        # first `from transformers import AutoImageProcessor` fails
        # inside the bundle with a misleading "Could not import module
        # 'AutoImageProcessor'" (real cause: hf_hub → requests chain).
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
        "tokenizers",
        "sklearn",  # transformers touches sklearn.utils in some paths
        "fsspec",
        "typing_extensions",
        # Our own packages need to ship at importable tree locations.
        "backend",
        "ml",
        "desktop",
    ],
    "includes": [
        "sqlite3",
        "queue",
    ],
    "excludes": [
        # Huge chromium binaries we never use.
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtWebEngineWidgets",
        # Dev-only deps.
        "tkinter",
        "matplotlib.tests",
        "pytest",
        # Server path we skip.
        "backend.serve",
    ],
    "plist": {
        "LSMinimumSystemVersion": "12.0",
        "CFBundleIdentifier": "com.cs31.preview",
        "CFBundleName": "CS31-1-Rhinoplasty-Prediction-Studio-Mac",
        "CFBundleDisplayName": "CS31-1-Rhinoplasty-Prediction-Studio-Mac",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        # Allows camera / photos access if we ever want to pick from
        # the system photo library via Qt file dialog.
        "NSPhotoLibraryUsageDescription": "Load your pre-op photo",
    },
}

# Strip None iconfile so py2app doesn't complain.
PY2APP_OPTIONS = {k: v for k, v in PY2APP_OPTIONS.items() if v is not None}


setup(
    app=APP_ENTRY,
    data_files=DATA_FILES,
    options={"py2app": PY2APP_OPTIONS},
    setup_requires=["py2app"],
)
