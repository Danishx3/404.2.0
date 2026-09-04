@echo off
title Build Ocular AI Executable
cd /d "%~dp0"

echo ============================================================
echo   Building Standalone Windows Executable (.exe)
echo ============================================================

pip show pyinstaller >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo Building Ocular_AI.exe with PyInstaller...
pyinstaller --noconfirm --onedir --windowed --name "Ocular_AI" ^
  --add-data "face_landmarker.task;." ^
  --add-data "storage.py;." ^
  --add-data "blink_counter.py;." ^
  gui_app.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo   BUILD SUCCESSFUL!
    echo   Executable is located in: dist\Ocular_AI\Ocular_AI.exe
    echo ============================================================
) else (
    echo [Error] Build failed.
)

pause
