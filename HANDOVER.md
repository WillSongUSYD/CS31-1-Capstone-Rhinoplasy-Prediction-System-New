# CS31-1 Rhinoplasty Prediction Studio — Project Handover

**Document version:** 1.0
**Last updated:** 2026-05-20
**Applies to application release:** v1.1.0

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository](#2-repository)
3. [Part A — End-User Guide](#part-a--end-user-guide)
   - [A.1 What the Application Does](#a1-what-the-application-does)
   - [A.2 System Requirements](#a2-system-requirements)
   - [A.3 Installation](#a3-installation)
   - [A.4 First Launch — Model Download](#a4-first-launch--model-download)
   - [A.5 Using the Application](#a5-using-the-application)
   - [A.6 Photo Requirements and Tips](#a6-photo-requirements-and-tips)
   - [A.7 File Locations](#a7-file-locations)
   - [A.8 Troubleshooting](#a8-troubleshooting)
4. [Part B — Developer Guide](#part-b--developer-guide)
   - [B.1 Repository Structure](#b1-repository-structure)
   - [B.2 The Three Deployment Targets](#b2-the-three-deployment-targets)
   - [B.3 Development Environment Setup](#b3-development-environment-setup)
   - [B.4 ML Pipeline](#b4-ml-pipeline)
   - [B.5 Backend API](#b5-backend-api)
   - [B.6 Frontend](#b6-frontend)
   - [B.7 Tests](#b7-tests)
   - [B.8 Building the Desktop Applications](#b8-building-the-desktop-applications)
   - [B.9 Architecture Notes](#b9-architecture-notes)
   - [B.10 Data and Patient Privacy](#b10-data-and-patient-privacy)
5. [Part C — Appendix](#part-c--appendix)

---

## 1. Project Overview

**CS31-1 Rhinoplasty Prediction Studio** is a research prototype that predicts
the post-operative result of rhinoplasty (nose surgery) from a single
pre-operative side-profile photograph. Given a patient's "before" profile
photo, the system generates a predicted "after" image and a short description
of the geometric changes to the nose.

The project has three deployment targets:

| Target | Description | Primary user |
|---|---|---|
| **Desktop application** | Self-contained macOS and Windows apps with a graphical interface | Clinicians / end users |
| **Python ML pipeline** | Dataset indexing, model training, evaluation | Researchers / developers |
| **Web application** | FastAPI backend + React frontend for browser-based inference | Developers / demos |

This handover document is divided into **Part A** (for end users of the
desktop application) and **Part B** (for developers who will maintain or
extend the source code).

> **Status:** This is a research prototype. Predictions are AI-generated
> approximations intended to support discussion and visualisation. They are
> **not** a medical guarantee of surgical outcome.

---

## 2. Repository

**Repository address:**

```
https://github.com/WillSongUSYD/CS31-1-Capstone-Rhinoplasy-Prediction-System-New
```

- **Default branch:** `main`
- **Large files:** The trained LoRA model weights are stored using **Git LFS**.
  You must have Git LFS installed before cloning, otherwise the weights file
  will be a small text pointer instead of the real ~28 MB model.

**Cloning the repository:**

```bash
# Install Git LFS once per machine
git lfs install

# Clone the repository
git clone https://github.com/WillSongUSYD/CS31-1-Capstone-Rhinoplasy-Prediction-System-New.git
cd CS31-1-Capstone-Rhinoplasy-Prediction-System-New

# Confirm LFS files were pulled (should be ~28 MB, not a few hundred bytes)
git lfs pull
```

All application source code lives under the `CS31_Source_Code_TuohengZhang/`
directory. Unless stated otherwise, every command in Part B is run from inside
that directory.

---

# Part A — End-User Guide

This part is for people who will **install and use** the desktop application.
No programming knowledge is required.

## A.1 What the Application Does

1. You provide a **side-profile photo** of a patient taken before surgery.
2. The application analyses the nose region and generates a **predicted
   post-operative image**.
3. It shows the original and the prediction **side by side** for comparison.
4. It produces a short text summary of the predicted change to the nose.
5. You can **save** the predicted image to your computer.

## A.2 System Requirements

| | macOS | Windows |
|---|---|---|
| Operating system | macOS 12 (Monterey) or newer, **Apple Silicon** (M1/M2/M3/M4) | Windows 10 or 11, 64-bit |
| Memory (RAM) | 8 GB minimum | 16 GB recommended (8 GB minimum) |
| Free disk space | ~6 GB (app + downloaded AI model) | ~6 GB (app + downloaded AI model) |
| Internet | Required **once**, on first launch, to download the AI model (~4 GB) | Required **once**, on first launch, to download the AI model (~4 GB) |

## A.3 Installation

The application is distributed as a compressed `.zip` file, available from the
**Releases** page of the repository (see Section 2).

### macOS

1. Download the macOS `.zip` and double-click it to unzip.
2. Drag **`CS31-1-Rhinoplasty-Prediction-Studio.app`** into your
   **Applications** folder.
3. Because the app is not signed with a paid Apple Developer certificate,
   macOS will block it the first time. To approve it:
   - **Right-click** (or Control-click) the app icon and choose **Open**.
   - A warning appears saying the developer cannot be verified. Click
     **Done** / **Cancel**.
   - Open **System Settings → Privacy & Security**.
   - Scroll down to the Security section. You will see a message such as
     *"CS31-1-Rhinoplasty-Prediction-Studio was blocked..."*. Click
     **Open Anyway**.
   - Confirm once more and enter your Mac password if prompted.
4. The app now opens. This approval is only needed **once**.

### Windows

1. Download the Windows `.zip` and unzip it to any location (for example, your
   Desktop).
2. Open the unzipped folder. It contains:

   | Item | Purpose |
   |---|---|
   | `CS31-1-Rhinoplasty-Prediction-Studio.exe` | The application. Double-click to run. |
   | `download_sd_model_v3.bat` | Backup AI-model downloader (see Troubleshooting). |
   | `README.txt` | Quick reference. |
   | `FIRST_LAUNCH.txt` | First-launch and download instructions. |
   | `_internal\` | Program files. **Do not delete, move or rename.** |

3. Keep **all** of these items together in the same folder. The application
   will not start if the `_internal` folder is missing.
4. Double-click **`CS31-1-Rhinoplasty-Prediction-Studio.exe`** to run.
   Windows SmartScreen may show a warning the first time — click
   **More info → Run anyway**.

## A.4 First Launch — Model Download

The first time the application starts, it must download a **~4 GB AI model**
(Stable Diffusion 1.5 Inpainting). This happens **only once**. Every launch
afterwards starts immediately.

1. Make sure you have a **stable internet connection**.
2. Start the application.
3. An onboarding window appears and begins the download.
4. **Keep the window open** until the download finishes. This typically takes
   **10–30 minutes**, depending on your connection.
5. When the download completes, the application is ready to use.

If the in-app download fails, see [A.8 Troubleshooting](#a8-troubleshooting).

## A.5 Using the Application

1. **Load a photo** — drag a pre-operative side-profile photo into the
   application window, or use the file picker.
2. **Prediction runs automatically** — once a valid photo is loaded, the
   application begins generating the prediction. A progress indicator is
   shown while it works.
3. **Review the result** — the original and the predicted post-operative
   image are displayed side by side. A short text description summarises the
   predicted change.
4. **Save** — use the Save button to write the predicted image to your
   computer. On macOS the default location is
   `~/Pictures/CS31-1-Rhinoplasty-Prediction-Studio/`.

## A.6 Photo Requirements and Tips

Prediction quality depends heavily on the input photo. Follow these guidelines.

**Required:**

- A **true side profile** — the face viewed at 90° from the front.
- **One person only** in the photo.
- File format: **JPEG, PNG, or WEBP**.
- Minimum size: **512 × 512 pixels**.

**Tips for the best results:**

- **Frame from the shoulders up.** Do not include the chest, the full body, or
  large amounts of clothing. Extra body area pulls the model's focus away from
  the face and produces **blurred predictions**.
- **Use a plain, uncluttered background** (a solid wall works well). Busy
  backgrounds add visual noise that lowers prediction quality.
- **Use the highest-resolution photo available.** The prediction is generated
  at the same resolution as the input, so a sharper photo produces a sharper,
  clearer result. 1024 pixels or larger is noticeably better than the 512-pixel
  minimum.
- Make sure the **nose is clearly visible** and well lit.

## A.7 File Locations

| Item | macOS | Windows |
|---|---|---|
| Downloaded AI model | `~/Library/Application Support/CS31-1-Rhinoplasty-Prediction-Studio/models/sd_base/inpaint/` | `%APPDATA%\CS31-1-Rhinoplasty-Prediction-Studio\models\sd_base\inpaint\` |
| Log file | `~/Library/Application Support/CS31-1-Rhinoplasty-Prediction-Studio/cs31-rhinoplasty-prediction-studio.log` | `%APPDATA%\CS31-1-Rhinoplasty-Prediction-Studio\cs31-rhinoplasty-prediction-studio.log` |
| Saved predictions | `~/Pictures/CS31-1-Rhinoplasty-Prediction-Studio/` | `~/Pictures/CS31-1-Rhinoplasty-Prediction-Studio/` |

> On Windows, paste the path (including `%APPDATA%`) into the File Explorer
> address bar to open the folder directly.

## A.8 Troubleshooting

| Problem | Cause and solution |
|---|---|
| **macOS: "app cannot be opened" / "unidentified developer"** | Expected for an unsigned app. Follow the right-click → Open approval steps in [A.3](#a3-installation). This is only needed once. |
| **Windows: SmartScreen warning** | Expected for an unsigned app. Click **More info → Run anyway**. |
| **Windows: app does not start at all** | The `_internal` folder is probably missing or was separated from the `.exe`. Re-unzip the original download and keep every item together. |
| **First-launch model download fails or stalls (Windows)** | Close the app. Double-click **`download_sd_model_v3.bat`** in the application folder. It finds or installs a compatible Python, creates an isolated environment, and downloads the model directly. The command window may look frozen for up to 60 seconds while it starts — this is normal. When it prints *"Download complete"*, close it and restart the app. |
| **First-launch model download fails (macOS)** | Confirm you have a stable internet connection, then quit and reopen the app to retry the download. Check the log file (see [A.7](#a7-file-locations)) for the specific error. |
| **Predictions look blurry** | Almost always a photo-quality issue. Re-shoot following [A.6](#a6-photo-requirements-and-tips): frame tightly on the face, use a plain background, and use a high-resolution image. |
| **Application is slow** | Inference is computationally heavy. On Windows it runs on the CPU; on Apple Silicon Macs it uses the GPU (MPS). Close other heavy applications and allow extra time per prediction. |
| **Something else went wrong** | Open the **log file** (see [A.7](#a7-file-locations)) — it records the detailed error message. Include this file when reporting an issue. |

---

# Part B — Developer Guide

This part is for developers who will maintain, rebuild, or extend the project.

## B.1 Repository Structure

All source code is under `CS31_Source_Code_TuohengZhang/`:

```
CS31_Source_Code_TuohengZhang/
├── ml/              ML pipeline: dataset indexing, training, evaluation, models
├── backend/         FastAPI API, inference dispatch, SQLite history
├── frontend/        React + Vite web application
├── desktop/         PyQt6 desktop app + build scripts (macOS & Windows)
├── data/            Manifests, splits, annotation templates (no patient images)
├── models/          Trained checkpoints and LoRA weights
├── artifacts/        Prepared image pairs and prediction outputs (gitignored)
├── reports/         Report drafts and review templates
├── tests/           pytest test suite
└── requirements.txt Python dependencies
```

The `.github/workflows/` directory (at the repository root) holds the
automated Windows build pipeline.

## B.2 The Three Deployment Targets

1. **Python ML pipeline** (`ml/`) — builds the dataset, trains models, and
   evaluates them. Command-line only.
2. **Web application** (`backend/` + `frontend/`) — a FastAPI server that
   performs inference, with a React browser UI.
3. **Desktop application** (`desktop/`) — a self-contained PyQt6 app, packaged
   as a macOS `.app` (via py2app) and a Windows folder/`.exe` (via PyInstaller).

All three share the same model and inference code in `ml/` and `backend/`.

## B.3 Development Environment Setup

Run all commands from `CS31_Source_Code_TuohengZhang/`.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows

# Install Python dependencies
pip install -r requirements.txt

# InsightFace is needed for nose-region detection but is not in
# requirements.txt — install it separately
pip install insightface
```

**Python version:** 3.12 is the supported version (used by both desktop builds).

The Stable Diffusion training/evaluation code additionally needs the Hugging
Face stack (`diffusers`, `transformers`, `peft`, `safetensors`, `accelerate`),
which is **not** in `requirements.txt` — install it separately when needed.

## B.4 ML Pipeline

```bash
# Build manifest.csv from the source images
python -m ml.index_dataset

# Produce 256-px image pairs and nose-ROI crops
python -m ml.prepare_pairs

# Train a model (full-face 256-px mode)
python -m ml.train_outcome --model pix2pix --epochs 30

# Quick smoke test
python -m ml.train_outcome --model pix2pix --epochs 1 --limit 32

# Evaluate
python -m ml.evaluate_outcome --model pix2pix --limit 16

# Stable Diffusion + LoRA training / evaluation
python -m ml.train_sd_inpaint --base <sd_base_dir> --out <lora_out> --steps 5000
python -m ml.evaluate_sd_inpaint --base <sd_base_dir> --lora <lora_dir> --out <out>
```

Append `_nose` to a model name (e.g. `pix2pix_nose`) to train in nose-ROI mode
(128 × 128) instead of full-face mode (256 × 256).

## B.5 Backend API

```bash
python -m backend.serve     # FastAPI on http://127.0.0.1:8000
```

CORS origins default to `localhost:5173`; override with the
`CS31_CORS_ORIGINS` environment variable.

## B.6 Frontend

```bash
cd frontend
npm install
npm run dev      # Vite dev server on http://localhost:5173
npm run build    # production build into frontend/dist/ (served by FastAPI at /)
```

## B.7 Tests

```bash
pytest tests/                                  # run all tests
pytest tests/test_dataset_tools.py -v          # run one file
```

## B.8 Building the Desktop Applications

### macOS (py2app)

```bash
pip install -r desktop/requirements-desktop.txt

# Preferred: build script handles rpath patching, code-signing and zipping
bash desktop/scripts/build_app.sh        # run from CS31_Source_Code_TuohengZhang/

# Run without bundling (for development)
python -m desktop
```

The V6 LoRA weights file must exist before building; `build_app.sh` checks for
it and fails early if missing.

### Windows (PyInstaller)

The Windows app is normally built **automatically** by GitHub Actions, but it
can also be built locally on a Windows machine:

```bat
desktop\scripts\build_windows.bat
```

This installs dependencies, runs PyInstaller against `desktop/CS31_windows.spec`,
assembles the distribution files, verifies the bundle, and produces a `.zip`.

**Automated build (recommended):**

1. Go to the repository on GitHub → **Actions** tab.
2. Select the **Build Windows App** workflow.
3. Click **Run workflow** → **Run workflow**.
4. When it finishes (~12 minutes), download the `.zip` from the run's
   **Artifacts** section.

The workflow also triggers automatically when a version tag (`v*`) is pushed.

**Windows distribution layout:** the build keeps the top level of the zip
clean — only the `.exe`, `download_sd_model_v3.bat`, `README.txt`, and
`FIRST_LAUNCH.txt` are visible; all runtime files (DLLs, Python, model data)
are inside the `_internal/` folder.

## B.9 Architecture Notes

**ML models** (`ml/models/`) — five model types share a common dispatcher in
`ml/runtime.py`: `autoencoder`, `pix2pix`, `cyclegan`, `diffusion`, and
`sd_inpaint_nose` (Stable Diffusion 1.5 Inpainting + LoRA). The production
desktop app uses the `sd_inpaint_nose` path.

**Device selection** — `ml/runtime.py:get_device` returns the first available
of **MPS** (Apple Silicon GPU) → **CUDA** → **CPU**. Checkpoints are always
saved to CPU so they load on any platform.

**Inference flow** — `backend/inference.py:run_prediction` splits the paired
image, loads the model (LRU-cached), runs inference, pastes the nose region
back for `_nose` models, and computes error metrics. Stable Diffusion
predictions are dispatched to `backend/inference_sd.py`.

**Nose ROI** — `ml/nose_roi.py` uses InsightFace (buffalo_l ONNX) for landmark
detection, with CLAHE-enhanced and proportional-heuristic fallbacks. The nose
mask is a tilted ellipse anchored between the eye midpoint and the nose tip.

**Desktop runtime paths** — `desktop/core/paths.py` resolves bundled,
read-only resources via `sys._MEIPASS` (PyInstaller) or the `.app` Resources
directory (py2app), and user-writable state under the platform application-data
directory. `desktop/app.py` calls `desktop.core.config.install_environment()`
before any backend/ml import, because those modules read environment variables
at import time.

**Stable Diffusion base model** — the ~4 GB SD 1.5 Inpainting base weights are
**not** committed to the repository. The desktop app downloads them on first
launch; for development they can be placed at `models/sd_base/inpaint/` and
pointed to via the `CS31_SD_BASE_DIR` environment variable. The much smaller
V6 LoRA weights **are** bundled with the app (via Git LFS in the repo).

## B.10 Data and Patient Privacy

**Patient images and identifying metadata must never be committed to Git.**
The following are excluded via `.gitignore` and contain patient face images or
personally identifiable information (PII):

- `CS31_Rhioplasty_Outcome_Prediction/` — raw dataset images
- `data/manifest.csv` — filename metadata
- `artifacts/dataset/` — derived image artifacts

When handing the project over, transfer the patient dataset through a secure,
private channel — **not** through the public repository.

---

# Part C — Appendix

## C.1 Quick Reference — Key Locations

| What | Where |
|---|---|
| Repository | `https://github.com/WillSongUSYD/CS31-1-Capstone-Rhinoplasy-Prediction-System-New` |
| Source code root | `CS31_Source_Code_TuohengZhang/` |
| Windows build pipeline | `.github/workflows/build-windows.yml` |
| macOS build script | `desktop/scripts/build_app.sh` |
| Windows build script | `desktop/scripts/build_windows.bat` |
| Windows PyInstaller spec | `desktop/CS31_windows.spec` |
| End-user docs shipped in the app | `desktop/dist_files/README.txt`, `desktop/dist_files/FIRST_LAUNCH.txt` |

## C.2 Version History

| Version | Highlights |
|---|---|
| v1.1.0 | First official Windows release; redesigned UI; fixed LoRA crash; fixed model-download server; platform-specific data paths. |
| v1.0.0 | Initial macOS release (previously named "CS31Preview"). |

## C.3 Known Limitations

- Predictions are AI-generated approximations, not a medical guarantee.
- Windows inference runs on the CPU; expect each prediction to take longer
  than on an Apple Silicon Mac.
- The first-launch model download requires a stable internet connection
  (~4 GB).
- The desktop apps are not signed with paid developer certificates, so the
  operating system shows a one-time security warning on first launch.

## C.4 Handover Notes

- The GitHub repository is owned by the **WillSongUSYD** account. To push
  changes, a maintainer needs write access to that repository.
- The trained LoRA weights are tracked with **Git LFS** — ensure Git LFS is
  installed before cloning or pushing.
- The `main` branch is the source of truth. The Windows build pipeline runs
  from `main`.
