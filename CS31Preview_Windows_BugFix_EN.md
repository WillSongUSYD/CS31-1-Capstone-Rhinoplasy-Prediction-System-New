# CS31Preview Windows — Developer Bug Report & Fix Summary

**Date:** 2026-05-19
**Branch / Commits:** `main` — `d749b58`, `d8a542a`
**Action required:** Pull latest commits and rebuild `CS31Preview.exe`

---

## Background

The Windows executable `CS31Preview.exe` was compiled from a macOS-first codebase. Two bugs were discovered when running the app on Windows for the first time. Both are already fixed in the source code (commits `d749b58` and `d8a542a` on the `main` branch). **The developer only needs to pull the latest commits and rebuild the Windows exe.**

---

## Bug 1 — App always shows the "Download Model" dialog even after the model is already downloaded

### Root Cause

`desktop/core/paths.py`, function `user_support_dir()`, was hardcoded to return the macOS path on **all** platforms:

```python
# OLD — WRONG on Windows
d = Path.home() / "Library" / "Application Support" / _APP_NAME
```

On Windows, `Path.home()` returns `C:\Users\<username>`. So the old code computed:

```
C:\Users\<username>\Library\Application Support\CS31Preview\
```

This directory does not exist in a standard Windows environment. The exe actually writes all its data to the Windows standard location:

```
C:\Users\<username>\AppData\Roaming\CS31Preview\    (i.e., %APPDATA%\CS31Preview\)
```

Because the model existence check (`is_sd_base_present()`) was looking in the wrong folder, it always returned `False` — so the download dialog appeared on every launch, even when the model was fully downloaded.

### Fix Applied — `desktop/core/paths.py`

```python
import platform
import os

def user_support_dir() -> Path:
    """Platform-appropriate writable app-data directory. Created on first access.

    - Windows: %APPDATA%\\CS31Preview\\
    - macOS:   ~/Library/Application Support/CS31Preview/
    """
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        d = Path(appdata) / _APP_NAME if appdata else Path.home() / _APP_NAME
    else:
        d = Path.home() / "Library" / "Application Support" / _APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d
```

### Impact After Rebuilding

- **New users:** The correct data directory `%APPDATA%\CS31Preview\` will be used from the first launch. No action needed.
- **Existing users who already downloaded the model to the wrong path** (`C:\Users\<username>\Library\Application Support\CS31Preview\models\sd_base\inpaint\`) will need to move the model folder manually to the correct location: `%APPDATA%\CS31Preview\models\sd_base\inpaint\`.

---

## Bug 2 — In-app download gets stuck at 0% (never starts)

### Root Cause

`huggingface_hub >= 0.25.0` introduced a CDN URL validation check. When the download endpoint is set to `https://hf-mirror.com` (a China-based mirror), the LFS binary files are served from a CDN domain that is **not** `*.huggingface.co`. The new validation rejects these URLs with the following error:

```
FileMetadataError: Distant resource does not seem to be on huggingface.co.
It is possible that a configuration issue prevents you from downloading
resources from https://huggingface.co.
```

The in-app downloader (`desktop/core/downloader.py`) sets `HF_ENDPOINT = https://hf-mirror.com` at runtime. If the build environment had `huggingface_hub >= 0.25` installed, PyInstaller bundles that version into the exe — and the download silently fails at 0%.

### Fix Applied — `desktop/requirements-desktop.txt`

```
# >=0.25 introduced a CDN-URL validation that rejects hf-mirror.com LFS files.
# Pin below that check; 0.21.x is the last version compatible with mirror
# endpoints and the diffusers/transformers versions used in this project.
huggingface_hub>=0.19,<0.25
```

### Critical Instruction for the Developer

When setting up the build environment to compile the new exe, run:

```bash
pip install -r desktop/requirements-desktop.txt
```

This must install `huggingface_hub` at version **0.19.x – 0.24.x** (NOT 0.25 or above). Verify with:

```bash
pip show huggingface-hub
# Expected output:  Version: 0.2x.x  (where the minor version is less than 25)
```

If the build machine already has `huggingface_hub >= 0.25` installed globally, the pinned requirement file will downgrade it within the project venv. **Do not override or ignore this version pin.**

---

## Additional Files (Non-exe — Already in Repo)

These files are complete as-is and require no further code changes. Include them alongside `CS31Preview.exe` in the Windows distribution zip:

| File | Purpose |
|---|---|
| `desktop/download_sd_model.bat` | One-click script for users to download the ~4 GB model before first launch. Auto-detects Python 3.9–3.13 via the `py` launcher; auto-installs Python 3.12 via `winget` if no compatible version is found; uses an isolated venv at `%APPDATA%\CS31Preview\download_env`; downloads from `huggingface.co` directly |
| `desktop/FIRST-LAUNCH.txt` | User-facing setup guide; now includes a Windows section at the top explaining the required pre-launch download step |

---

## Summary: Steps for the Developer

| Step | Command / Action |
|---|---|
| 1. Pull latest source | `git pull` — gets commits `d749b58` and `d8a542a` |
| 2. Set up build venv | `pip install -r desktop/requirements-desktop.txt` |
| 3. Verify huggingface_hub version | `pip show huggingface-hub` → must be `< 0.25` |
| 4. Rebuild exe | Run PyInstaller from the updated source |
| 5. Package distribution | Include `CS31Preview.exe`, `download_sd_model.bat`, and `FIRST-LAUNCH.txt` in the zip |

---

*Generated 2026-05-19. All source changes are committed to the `main` branch.*
