#!/usr/bin/env pythonw
"""
Ocular AI - Windows Windowed Launcher (No Console)
-------------------------------------------------
Executed by pythonw.exe to launch the desktop application cleanly
without displaying a background terminal window.
"""

import os
import sys

# Ensure current directory is in path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import gui_app

if __name__ == "__main__":
    gui_app.main()
