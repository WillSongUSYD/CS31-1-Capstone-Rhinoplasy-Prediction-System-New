@echo off
setlocal

cd /d "%~dp0"

echo == CS31Preview Windows executable build ==
echo Project: %cd%

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment ...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo Failed to create virtual environment. Please install Python 3.9 or newer.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

echo Installing/updating dependencies ...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r desktop\requirements-desktop.txt
if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo Building Windows executable with PyInstaller ...
python -m PyInstaller desktop\CS31Preview_windows.spec --noconfirm --clean
if errorlevel 1 (
    echo PyInstaller build failed.
    pause
    exit /b 1
)

echo Creating zip package ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path 'dist\CS31Preview_Windows.zip') { Remove-Item 'dist\CS31Preview_Windows.zip' -Force }; Compress-Archive -Path 'dist\CS31Preview' -DestinationPath 'dist\CS31Preview_Windows.zip' -Force"
if errorlevel 1 (
    echo Zip packaging failed.
    pause
    exit /b 1
)

echo Done. Output: dist\CS31Preview_Windows.zip
pause
