# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CS31 is a research prototype for paired pre-op / post-op rhinoplasty outcome prediction. It has three deployment targets:
1. **Python ML pipeline** — dataset indexing, training, evaluation
2. **FastAPI + React web app** — browser-based inference UI
3. **PyQt6 macOS desktop app** — self-contained `.app` bundle (CS31Preview)

All source code lives under `CS31_Source_Code_TuohengZhang/`. The packaged `.app` and zip are sibling items at the repo root.

**Patient data / PII**: `CS31_Rhioplasty_Outcome_Prediction/` (raw images) and `data/manifest.csv` are in `.gitignore` and must never be committed — they contain patient face images and filename metadata that constitute PII. All derived image artifacts under `artifacts/dataset/` are similarly excluded.

## Commands

All commands run from `CS31_Source_Code_TuohengZhang/` unless noted.

### Python environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install insightface   # not in requirements.txt; needed for nose ROI (buffalo_l ONNX)
```

### ML pipeline
```bash
python -m ml.index_dataset           # build manifest.csv from source images
python -m ml.prepare_pairs           # produce 256-px pairs + nose ROI crops
python -m ml.train_outcome --model pix2pix --epochs 30
python -m ml.train_outcome --model pix2pix --epochs 1 --limit 32   # smoke test
python -m ml.train_sd_inpaint --base <sd_base_dir> --out <lora_out> --steps 5000
python -m ml.evaluate_outcome --model pix2pix --limit 16
python -m ml.evaluate_sd_inpaint --base <sd_base_dir> --lora <lora_dir> --out <out>
```

### Backend API
```bash
python -m backend.serve              # FastAPI on http://127.0.0.1:8000
```
CORS origins default to `localhost:5173`; override via `CS31_CORS_ORIGINS` env var.

### Frontend
```bash
cd frontend
npm install
npm run dev          # Vite dev server on http://localhost:5173
npm run build        # output to frontend/dist/ (served by FastAPI at /)
```

### Tests
```bash
pytest tests/                                                          # all tests
pytest tests/test_dataset_tools.py -v                                  # one file
pytest tests/test_dataset_tools.py::test_split_paired_image_halves_have_equal_size  # one test
```

### Desktop app (macOS only)
```bash
pip install -r desktop/requirements-desktop.txt
# Preferred: uses build script that handles rpath patching + code-signing + zip
bash desktop/scripts/build_app.sh          # must run from CS31_Source_Code_TuohengZhang/

# Or directly:
python desktop/setup.py py2app --arch arm64
python -m desktop                          # run without bundling
```
The V6 LoRA (`models/outcome_v3_512/sd_inpaint_nose_v6/step_10000/pytorch_lora_weights.safetensors`) must exist before building; `build_app.sh` checks this and fails early if missing. The script also fixes torchvision rpath issues in the bundle and re-signs affected `.so` files.

## Architecture

### Key paths (`ml/config.py`)
| Constant | Default path |
|---|---|
| `SOURCE_DIR` | `CS31_Rhioplasty_Outcome_Prediction/` (raw dataset ZIP files) |
| `PAIR_256_DIR` | `artifacts/dataset/pairs_256/` |
| `PAIR_ALIGNED_DIR` | `artifacts/dataset/pairs_aligned_256/` |
| `MASK_DIR` | `artifacts/dataset/masks_256/` |
| `MODELS_DIR` | `models/` |
| `PREDICTIONS_DIR` | `artifacts/predictions/` (overrideable via `CS31_PREDICTIONS_DIR`) |
| `DB_PATH` | `data/history.sqlite3` |

### ML models (`ml/models/`)
Four architectures share the `create_model` / `model_output` dispatcher in `ml/runtime.py`:
- **autoencoder** — simple encoder-decoder
- **pix2pix** — UNet generator + PatchGAN discriminator
- **cyclegan** — ResNet generator pair + PatchGAN discriminators
- **diffusion** — `DiffusionFeasibilityModel` (DDPM-style feasibility scorer)

`ml/models/common.py` houses shared building blocks: the PatchGAN discriminator, norm-layer helpers, and `set_requires_grad`.

Each model can be trained in full-face mode (256×256) or nose-ROI mode (128×128) by appending `_nose` to the model name (e.g. `pix2pix_nose`). The `_nose` suffix is stripped by `_base_model_name()` before instantiation.

The fifth model type `sd_inpaint_nose` is a Stable Diffusion 1.5 Inpainting + LoRA pipeline (Hugging Face `diffusers`); it uses a completely separate code path in `ml/models/sd_inpaint.py` and `ml/train_sd_inpaint.py`. `diffusers` and `transformers` are not in `requirements.txt` — install them separately for SD training/evaluation.

### Device selection
`ml/runtime.py:get_device` returns the first available: **MPS** (Apple Silicon) → **CUDA** → **CPU**. Checkpoints are saved to CPU to avoid device pinning across platforms.

### Checkpoints
`ml/runtime.py:save_checkpoint` writes two sibling files:
- `models/outcome/<model_name>/best.pt` — raw `state_dict` (weights-only safe)
- `models/outcome/<model_name>/best.meta.json` — training metadata sidecar (epoch, image_size, history)

`load_model_from_checkpoint` tries `weights_only=True` first; falls back to legacy pickled load with a loud warning. The inference layer hardcodes `checkpoint_name = "best.pt"` — training saves both `latest.pt` (every epoch) and `best.pt` (best val L1).

### Backend / inference flow
1. `backend/serve.py` — FastAPI app; validates model name against `VALID_MODEL_NAMES` allowlist, off-loads PIL/GPU work to a threadpool.
2. `backend/inference.py:run_prediction` — main dispatch. Splits paired images, loads model via LRU cache (`_MODEL_CACHE_MAX=4`), runs inference, pastes nose back for `_nose` models, computes paired MSE. Then calls `ml/landmarks.py:detect_landmarks` (MediaPipe) for nose geometry and `ml/description.py:generate_description` to produce a human-readable surgical change summary.
3. For `sd_inpaint_*` models, dispatches to `_run_sd_prediction` → `backend/inference_sd.py` (deferred import to avoid ~800MB diffusers load at startup). Mask-weighted MSE is computed on the nose region only.
4. `backend/db.py` — SQLite history via `data/history.sqlite3`.

### Desktop app architecture
`desktop/app.py` must call `desktop.core.config.install_environment()` **before any backend/ml import** because those modules read env vars at import time. Key env vars set:
- `CS31_PREDICTIONS_DIR` → `~/Library/Application Support/CS31Preview/predictions/`
- `CS31_SD_BASE_DIR` / `CS31_SD_LORA_DIR` → bundled model paths
- `HF_ENDPOINT` — offline mode or mirror

The desktop uses `desktop/core/inference_worker.py` (runs inference in a QThread) and `desktop/main_window.py` / `desktop/widgets/` for the PyQt6 UI. `desktop/core/image_geometry.py` handles letterbox/padding transforms (fit arbitrary-aspect images to a square canvas and restore them after inference). The app bundles the V6 LoRA weights but downloads the SD 1.5 base (~4 GB) on first launch via `desktop/widgets/onboarding_dialog.py`.

### Nose ROI pipeline
`ml/nose_roi.py` uses InsightFace (buffalo_l ONNX) for 5-point landmark detection with a CLAHE-enhanced fallback and a proportional heuristic fallback. The nose mask is a tilted ellipse anchored on `eye_mid → nose_tip`. The ellipse constants were tuned through V3–V6 to avoid covering eyebrows while fully covering columella and nostrils.

A single call to `get_nose_roi_box` is made per prediction and the box is reused for both extract and paste-back to keep the two operations geometrically consistent.

### Dataset data flow
`data/manifest.csv` + `data/splits.csv` drive all training. `ml/data.py:load_pairs` joins them, filters duplicates and frontal views, returns `PairItem` objects. Training images are stored as single-canvas paired images (left=pre, right=post); `ml/dataset_tools.py:split_paired_image` splits them. For square inputs the split is top/bottom; aspect ratio determines orientation.

`data/cases.csv`, `data/notes.csv`, and `data/annotation_template.csv` hold scaffolded cost/NLP schemas — not used for formal modeling until real labels are added.

### SD base model paths
SD 1.5 Inpainting base weights are stored at `models/sd_base/inpaint/` (not bundled in the repo; ~3.5 GB). Override via `CS31_SD_BASE_DIR`. LoRA output goes to `models/outcome_v3_512/<model_name>/`.
