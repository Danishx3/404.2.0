@echo off
title Ocular AI Launcher
cd /d "%~dp0"

:: Check if python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [Error] Python is not installed or not found in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Launch with pythonw (no terminal) if available, otherwise python
where pythonw >nul 2>nul
if %ERRORLEVEL% equ 0 (
    start "" pythonw.exe "%~dp0app.pyw"
) else (
    start "" python.exe "%~dp0gui_app.py"
)

exit /b 0
