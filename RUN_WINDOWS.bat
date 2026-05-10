@echo off
setlocal

cd /d "%~dp0"

echo == CS31Preview Windows runner ==
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

echo Starting CS31Preview ...
python -m desktop
pause
