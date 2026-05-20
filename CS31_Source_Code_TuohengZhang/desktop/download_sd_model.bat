@echo off
setlocal
title CS31-1-Rhinoplasty-Prediction-Studio - Setup and Download

set "MODEL_DIR=%APPDATA%\CS31-1-Rhinoplasty-Prediction-Studio\models\sd_base\inpaint"
set "VENV_DIR=%APPDATA%\CS31-1-Rhinoplasty-Prediction-Studio\download_env"
set PYTHON=

echo CS31-1-Rhinoplasty-Prediction-Studio - Setup and Download
echo ================================================
echo.

:: ── 1. Check if model already installed ──────────────────────────────────────
if exist "%MODEL_DIR%\unet\diffusion_pytorch_model.bin" goto :already_done
if exist "%MODEL_DIR%\unet\diffusion_pytorch_model.safetensors" goto :already_done

echo Model not found - download required (~4 GB, 10-30 minutes).
echo Do NOT close this window once download starts.
echo.

:: ── 2. Find a compatible Python (3.9-3.13) ───────────────────────────────────
echo Looking for a compatible Python version (3.9 - 3.13)...

:: Prefer 3.12 via the py launcher, then fall back through other versions
py -3.12 --version >nul 2>&1 && set "PYTHON=py -3.12"
if not defined PYTHON ( py -3.11 --version >nul 2>&1 && set "PYTHON=py -3.11" )
if not defined PYTHON ( py -3.13 --version >nul 2>&1 && set "PYTHON=py -3.13" )
if not defined PYTHON ( py -3.10 --version >nul 2>&1 && set "PYTHON=py -3.10" )
if not defined PYTHON ( py -3.9  --version >nul 2>&1 && set "PYTHON=py -3.9"  )

:: Fall back to bare 'python' if its version is in range (using Python itself to check)
if not defined PYTHON (
    python -c "import sys; exit(0 if (3,9) <= sys.version_info < (3,14) else 1)" >nul 2>&1
    if not errorlevel 1 set PYTHON=python
)

:: No compatible Python found - try winget auto-install
if not defined PYTHON (
    echo No compatible Python found. Attempting to install Python 3.12 automatically...
    echo You may see a permissions prompt - click Yes to allow the installation.
    echo.
    winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
    if not errorlevel 1 (
        echo Python 3.12 installed successfully.
        py -3.12 --version >nul 2>&1 && set "PYTHON=py -3.12"
    )
)

if not defined PYTHON (
    echo.
    echo ERROR: Could not find or install a compatible Python ^(3.9 - 3.13^).
    echo.
    echo Please install Python 3.12 manually:
    echo   https://www.python.org/downloads/release/python-3119/
    echo During installation, check "Add Python to PATH".
    echo Then double-click this script again.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('%PYTHON% --version 2^>^&1') do echo Using Python %%i
echo.

:: ── 3. Create isolated virtual environment (if not already done) ──────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating isolated environment at:
    echo   %VENV_DIR%
    %PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Done.
    echo.
)

:: ── 4. Activate venv and install huggingface_hub ─────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"

echo Checking huggingface_hub and colorama...
python -m pip show huggingface-hub >nul 2>&1
if errorlevel 1 (
    echo Installing huggingface_hub and colorama...
    python -m pip install "huggingface_hub>=0.19" colorama
    python -m pip show huggingface-hub >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install huggingface_hub.
        echo Try running this manually to see the full error:
        echo   python -m pip install huggingface_hub colorama
        echo.
        pause
        exit /b 1
    )
) else (
    python -m pip show colorama >nul 2>&1
    if errorlevel 1 python -m pip install colorama >nul 2>&1
)
echo huggingface_hub OK.
echo.

:: ── 5. Download model ─────────────────────────────────────────────────────────
mkdir "%MODEL_DIR%" 2>nul

echo Saving model to:
echo   %MODEL_DIR%
echo.
echo -----------------------------------------------------------------------
echo  PLEASE WAIT - the window may appear frozen for up to 60 seconds
echo  while the download library initialises. Progress bars will appear
echo  once the connection is established. Do NOT close this window.
echo -----------------------------------------------------------------------
echo.

set HF_ENDPOINT=https://huggingface.co
set PYTHONUNBUFFERED=1
set TQDM_MININTERVAL=0.5
python -u -c "import os, sys; os.environ['HF_ENDPOINT']='https://huggingface.co'; print('[1/3] Loading download library...', flush=True); import colorama; colorama.init(); from huggingface_hub import snapshot_download; print('[2/3] Resolving file list from HuggingFace (may take ~30s)...', flush=True); snapshot_download(repo_id='botp/stable-diffusion-v1-5-inpainting', local_dir=r'%MODEL_DIR%', allow_patterns=['model_index.json','unet/*.bin','unet/config.json','vae/*.bin','vae/config.json','text_encoder/*.bin','text_encoder/*.json','tokenizer/*','scheduler/*','feature_extractor/*']); print('[3/3] All files saved.', flush=True)"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ====================================
    echo Download complete^^! You can now launch CS31-1-Rhinoplasty-Prediction-Studio.
    echo Model saved to:
    echo   %MODEL_DIR%
) else (
    echo.
    echo ====================================
    echo Download FAILED. See error message above.
    echo If the error says "not on huggingface.co", run this and try again:
    echo   "%VENV_DIR%\Scripts\python.exe" -m pip install "huggingface_hub==0.21.3"
)

goto :end

:already_done
echo Model already installed at:
echo   %MODEL_DIR%
echo.
echo Nothing to do. You can launch CS31-1-Rhinoplasty-Prediction-Studio now.

:end
echo.
pause
