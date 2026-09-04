# 👁️ Ocular AI — Blink & Yawn Ergonomics Tracker
### *Windows Desktop Application Edition*

An AI-powered ocular ergonomics and fatigue monitoring desktop application built in Python using **Google MediaPipe FaceLandmarker**, **CustomTkinter**, and **SQLite**.

---

## ✨ Features

- **🖥️ Modern Windows Desktop GUI**: Built with CustomTkinter featuring a sleek dark-mode interface, embedded live 30 FPS webcam feed, responsive cards, and multi-tab navigation.
- **💾 Local SQLite Data Storage**: Automatically logs every tracking session to `data/blink_history.db`. Displays session history, duration, blinks, yawns, and average BPM with one-click **Export to CSV**.
- **🎯 Precision Blink & Wink Tracking**: Uses neural blendshapes (`eyeBlinkLeft`, `eyeBlinkRight`) with hysteresis debouncing (40ms – 600ms) to eliminate false triggers.
- **🥱 Speech-Immune Yawn Engine**: Distinguishes true deep yawns from conversational speech by pairing geometric **Mouth Aspect Ratio (MAR)** with a 58% threshold and 1.3s continuous hold requirement.
- **😴 Alertness & Drowsiness Index**: Evaluates blink rate and yawn frequency to determine fatigue levels (`ALERT & ATTENTIVE`, `TIRED`, or `DROWSY - Take a break!`).
- **📈 Real-Time Gauges & Waveforms**: Live progress bars for Left Eye, Right Eye, and Mouth Openness with dynamic threshold markers.
- **🔊 Non-Blocking Audio Cues**: Distinct audio blips on blinks/winks and gentle chimes on yawns.
- **🚀 Native Windows Packaging**:
  - Desktop Shortcut on your Windows Desktop: **`Ocular AI - Blink & Yawn Tracker.lnk`**
  - **`app.pyw`**: Windowed mode without command prompt window.
  - **`Launch_App.bat`**: Instant one-click runner.
  - **`build_exe.bat`**: Compiles a standalone `.exe` with PyInstaller.

---

## 🚀 How to Launch the Application

### Option 1: Desktop Shortcut (Recommended)
Double-click the **`Ocular AI - Blink & Yawn Tracker`** shortcut on your Windows Desktop!

### Option 2: One-Click Batch Launcher
Double-click **`Launch_App.bat`** in this folder.

### Option 3: Terminal Command
```powershell
python gui_app.py
```
*(Or use `pythonw app.pyw` to run without a background terminal)*

---

## 📑 Application Tabs

1. **🖥️ Live Monitor**:
   - Live camera stream with optional mesh contours.
   - Large live counters for Blinks, Yawns, and Winks.
   - Real-time BPM gauge, fatigue indicator, and eye/mouth openness bars.
   - Interactive buttons: Pause/Resume Camera, Toggle Mesh, Toggle Sound, Reset Session.
2. **📊 History & Analytics**:
   - Lifetime summary cards (Lifetime Blinks, Lifetime Yawns, Total Hours Tracked, Average BPM).
   - Scrollable history table of past sessions.
   - **`Export to CSV`** button for Excel analysis.
3. **⚙️ Settings**:
   - Camera device index selector (0, 1, 2).
   - Yawn threshold & hold duration sliders.
   - Blink sensitivity sliders.
   - Audio and Auto-save preferences.

> **Note**: The neural model (`face_landmarker.task`) is already included in this repository. If run on a new computer, it will automatically download from Google's official CDN on first launch.

---

## 🎮 Keyboard Controls

| Key | Action |
|---|---|
| `R` | **Reset** blinks, yawns, winks, and session timers |
| `S` | **Toggle Sound** feedback (ON / OFF) |
| `M` | **Toggle Mesh** visualization (eyes, pupils, and neon lip contours) |
| `G` | **Toggle Graph** for real-time eyelid dynamics waveform |
| `Y` | **Toggle Yawn Detection** (ON / OFF) |
| `-` / `+` | **Decrease / Increase Yawn Sensitivity** threshold |
| `C` | **Auto-Calibrate** eye sensitivity (3-second calibration) |
| `H` | **Toggle Help** shortcuts overlay |
| `Q` / `ESC` | **Quit** the application |

---

## 🧠 How It Works

1. **Neural Inference**: Each camera frame is passed to MediaPipe's `FaceLandmarker` running on-device CPU inference.
2. **Hybrid Eye & Mouth Feature Extraction**:
   - **Blinks**: Neural blendshapes (`eyeBlinkLeft`, `eyeBlinkRight`) with hysteresis debouncing (40ms – 600ms).
   - **Yawns**: Ensemble of geometric **Mouth Aspect Ratio (MAR)** (inner lip distances) + neural `jawOpen` blendshape score.
3. **Speech-Immune Yawn Engine**:
   - **Speech Rejection**: Conversational speech (mouth opening ~15%–35%) never reaches the yawn threshold (`58%`).
   - **Continuous Hold Requirement**: To prevent syllables and vowels from accumulating, the mouth must be held wide open continuously (`>= 58%`) for **`>= 1.30 seconds`**.
   - **Instant Flutter Reset**: If the mouth moves or dips below 58% during speech articulation, the hold timer immediately resets to zero.
   - **Live Hold Countdown**: While yawning, the HUD displays a live progress counter: `😮 YAWNING... (0.8s / 1.3s)` so you can see the system validating the hold in real time.
4. **Ergonomic Rate & Fatigue Analysis**:
   - Rolling 60-second time-series buffer of blinks calculates real-time **BPM**.
   - Cross-analyzes yawns and blink rates to evaluate fatigue (`ALERT`, `TIRED`, `DROWSY`).
