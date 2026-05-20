@echo off
setlocal
title CS31-1-Rhinoplasty-Prediction-Studio - Windows Build
set "REPO=%~dp0..\.."

echo == CS31-1-Rhinoplasty-Prediction-Studio Windows build ==
echo repo: %REPO%
echo.

:: 1. Verify LoRA is present.
set "LORA=%REPO%\models\outcome_v3_512\sd_inpaint_nose_v6\step_10000\pytorch_lora_weights.safetensors"
if not exist "%LORA%" (
    echo ERROR: V6 LoRA not found at:
    echo   %LORA%
    echo Pull it from the remote training server first.
    pause & exit /b 1
)
for %%F in ("%LORA%") do echo   LoRA: %%~zF bytes

:: 2. Check Python.
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.12 from python.org.
    pause & exit /b 1
)
echo Using Python: & python --version

:: 3. Create venv if not present.
if not exist "%REPO%\.venv\Scripts\activate.bat" (
    echo Creating virtual environment ...
    python -m venv "%REPO%\.venv"
    if errorlevel 1 ( echo ERROR: venv creation failed. & pause & exit /b 1 )
)
call "%REPO%\.venv\Scripts\activate.bat"

:: 4. Install dependencies.
echo Installing dependencies (first run takes 10-20 minutes) ...
python -m pip install --upgrade pip wheel
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install fastapi==0.115.12 uvicorn==0.34.0 pydantic==2.11.1 python-multipart==0.0.20 jinja2==3.1.6 pandas==2.2.3 Pillow==11.1.0 ImageHash==4.3.1 scikit-image==0.24.0 matplotlib==3.9.4 tqdm==4.67.1 lpips==0.1.4 mediapipe scikit-learn
pip install diffusers transformers peft safetensors accelerate "huggingface_hub>=1.5.0"
pip install insightface onnxruntime
pip install PyQt6 pyinstaller

:: 5. Clean previous build.
if exist "%REPO%\dist" rmdir /s /q "%REPO%\dist"
if exist "%REPO%\build" rmdir /s /q "%REPO%\build"

:: 6. Build.
echo == PyInstaller build ==
cd /d "%REPO%"
python -m PyInstaller desktop\CS31_windows.spec
if errorlevel 1 ( echo BUILD FAILED. & pause & exit /b 1 )

:: 7. Verify.
echo == Verifying bundle ==
python desktop\scripts\verify_bundle_windows.py "dist\CS31-1-Rhinoplasty-Prediction-Studio"
if errorlevel 1 ( echo VERIFY FAILED. & pause & exit /b 1 )

:: 8. Zip.
echo == Zipping ==
set "OUTDIR=dist\CS31-1-Rhinoplasty-Prediction-Studio"
set "ZIPNAME=dist\CS31-1-Rhinoplasty-Prediction-Studio-Windows.zip"
if exist "%ZIPNAME%" del "%ZIPNAME%"
powershell -Command "Compress-Archive -Path '%OUTDIR%' -DestinationPath '%ZIPNAME%' -CompressionLevel Optimal"
if errorlevel 1 ( echo ZIP FAILED. & pause & exit /b 1 )

echo.
echo == Done ==
for %%F in ("%ZIPNAME%") do echo Zip: %%~nxF (%%~zF bytes)
echo.
echo Distribute %ZIPNAME%.
echo Users double-click CS31-1-Rhinoplasty-Prediction-Studio.exe inside the folder.
echo The app downloads the 4 GB SD model on first launch.
echo.
pause
