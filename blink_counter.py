#!/usr/bin/env python3
"""
Eye Blink Tracker AI
-------------------
A real-time eye blink counter and ocular health monitor using MediaPipe FaceLandmarker
and OpenCV with an interactive cyber-HUD interface.

Features:
- Accurate neural blendshape blink & wink tracking (eyeBlinkLeft, eyeBlinkRight)
- Eye Aspect Ratio (EAR) & facial landmark contour visualization
- Blinks Per Minute (BPM) & ocular fatigue / dry-eye stare alerts
- Real-time rolling graph of eyelid closure dynamics
- Audio feedback on blink (toggleable)
- Interactive calibration mode to adapt to glasses or varied eye shapes
- Semi-transparent glassmorphic HUD overlay
"""

import os
import sys
import time
import math
import queue
import urllib.request
import threading
from collections import deque
import cv2
import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Windows native sound support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ==============================================================================
# Constants & Model Configurations
# ==============================================================================
MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Face Mesh Landmark Indices for Eyes & Irises
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Default Blink Thresholds (Blendshapes range 0.0 = wide open, 1.0 = fully closed)
DEFAULT_CLOSE_THRESHOLD = 0.48
DEFAULT_OPEN_THRESHOLD = 0.22
MIN_BLINK_DURATION_MS = 60    # Ignore micro-jitters
MAX_BLINK_DURATION_MS = 600   # Avoid counting prolonged eyes-closed as repeated blinks
STARE_ALERT_SECONDS = 12.0    # Warn user about dry eyes if no blink for > 12s

# Palette (BGR)
COLOR_BG_CARD = (22, 25, 34)
COLOR_BORDER = (65, 75, 95)
COLOR_ACCENT_CYAN = (235, 206, 0)       # Vibrant Cyan in BGR
COLOR_ACCENT_GREEN = (110, 225, 60)     # Neon Green in BGR
COLOR_ACCENT_YELLOW = (40, 200, 245)    # Amber Yellow in BGR
COLOR_ACCENT_RED = (60, 70, 245)        # Soft Red in BGR
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_DIM = (160, 165, 180)


# ==============================================================================
# Model Downloader
# ==============================================================================
def ensure_model_asset(model_path=MODEL_FILENAME):
    """Ensure the FaceLandmarker task model asset is available locally."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        return model_path

    print(f"[ModelManager] Downloading '{model_path}' from Google Cloud CDN...")
    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading model: {percent}% [{count * block_size // 1024} KB / {total_size // 1024} KB]")
        sys.stdout.flush()

    urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=_progress)
    print("\n[ModelManager] Download completed successfully.")
    return model_path


# ==============================================================================
# Sound Feedback Manager
# ==============================================================================
class SoundFeedback:
    """Non-blocking audio feedback player using background daemon threads."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._queue = queue.Queue(maxsize=5)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            sound_type = self._queue.get()
            if self.enabled and HAS_WINSOUND:
                try:
                    if sound_type == 'blink':
                        winsound.Beep(1400, 35)  # Crisp high blip
                    elif sound_type == 'wink':
                        winsound.Beep(950, 45)   # Lower blip
                    elif sound_type == 'alert':
                        winsound.Beep(650, 120)  # Gentle warning tone
                except Exception:
                    pass
            self._queue.task_done()

    def play(self, sound_type='blink'):
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(sound_type)
        except queue.Full:
            pass

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled


# ==============================================================================
# Eye Blink & Ergonomics Tracker
# ==============================================================================
class BlinkTracker:
    """Tracks blink events, winks, blink rate (BPM), and eye fatigue."""
    def __init__(self, close_threshold=DEFAULT_CLOSE_THRESHOLD, open_threshold=DEFAULT_OPEN_THRESHOLD):
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold

        # Counters
        self.total_blinks = 0
        self.left_winks = 0
        self.right_winks = 0

        # State machine flags
        self.is_eyes_closed = False
        self.is_left_closed = False
        self.is_right_closed = False
        self.eye_closed_start_time = None
        self.last_blink_time = time.time()
        self.last_state = "OPEN"

        # Rolling statistics
        self.blink_timestamps = deque()          # Timestamps of blinks in the last 60s
        self.closure_history = deque(maxlen=90)  # Rolling values for sparkline graph
        self.session_start = time.time()
        self.pulse_frames = 0                    # Highlight animation frames for HUD

    def reset(self):
        """Reset all counters and statistics."""
        self.total_blinks = 0
        self.left_winks = 0
        self.right_winks = 0
        self.is_eyes_closed = False
        self.is_left_closed = False
        self.is_right_closed = False
        self.eye_closed_start_time = None
        self.last_blink_time = time.time()
        self.session_start = time.time()
        self.blink_timestamps.clear()
        self.closure_history.clear()
        self.pulse_frames = 0
        self.last_state = "OPEN"

    def update(self, left_score, right_score):
        """
        Process single-frame left and right eyelid closure scores (0.0 to 1.0).
        Returns: event string ("BLINK", "LEFT_WINK", "RIGHT_WINK", or None)
        """
        now = time.time()
        event = None

        if left_score is None or right_score is None:
            self.closure_history.append(0.0)
            return None

        # Record combined closure score for rolling graph
        avg_closure = (left_score + right_score) / 2.0
        self.closure_history.append(avg_closure)

        left_is_shut = left_score >= self.close_threshold
        right_is_shut = right_score >= self.close_threshold
        both_shut = left_is_shut and right_is_shut

        # Both eyes closed together (potential bilateral blink)
        if both_shut:
            # Cancel any unilateral wink tracking to avoid false winks during blinks
            self.is_left_closed = False
            self.is_right_closed = False
            if not self.is_eyes_closed:
                self.is_eyes_closed = True
                self.eye_closed_start_time = now
                self.last_state = "CLOSING"
        else:
            # Eyes re-opened (check if previous state was closed)
            if self.is_eyes_closed:
                # Eyelids must open past open_threshold
                if left_score < self.open_threshold and right_score < self.open_threshold:
                    duration_ms = (now - (self.eye_closed_start_time or now)) * 1000.0
                    self.is_eyes_closed = False
                    self.last_state = "OPEN"

                    if 40 <= duration_ms <= MAX_BLINK_DURATION_MS:
                        self.total_blinks += 1
                        self.last_blink_time = now
                        self.blink_timestamps.append(now)
                        self.pulse_frames = 9
                        event = "BLINK"
                elif left_score > 0.75 and right_score > 0.75:
                    # User is keeping eyes closed (prolonged close, don't count multiple times)
                    pass
                else:
                    # Partial opening, remain in closed state until fully opened
                    pass

        # Check for isolated winks (only if not in bilateral blink)
        if not self.is_eyes_closed:
            # Left Eye Wink
            if left_is_shut and not right_is_shut and right_score < self.open_threshold:
                if not self.is_left_closed:
                    self.is_left_closed = True
                    self.left_wink_start_time = now
            elif self.is_left_closed and left_score < self.open_threshold:
                wink_dur = (now - getattr(self, 'left_wink_start_time', now)) * 1000.0
                self.is_left_closed = False
                if 50 <= wink_dur <= 900:
                    self.left_winks += 1
                    event = "LEFT_WINK"

            # Right Eye Wink
            if right_is_shut and not left_is_shut and left_score < self.open_threshold:
                if not self.is_right_closed:
                    self.is_right_closed = True
                    self.right_wink_start_time = now
            elif self.is_right_closed and right_score < self.open_threshold:
                wink_dur = (now - getattr(self, 'right_wink_start_time', now)) * 1000.0
                self.is_right_closed = False
                if 50 <= wink_dur <= 900:
                    self.right_winks += 1
                    event = "RIGHT_WINK"

        # Clean up timestamps older than 60s for BPM
        cutoff = now - 60.0
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

        if self.pulse_frames > 0:
            self.pulse_frames -= 1

        return event

    def get_bpm(self):
        """Calculate Blinks Per Minute (BPM)."""
        elapsed = time.time() - self.session_start
        if elapsed < 1.0:
            return 0.0
        if elapsed < 60.0:
            # Extrapolate initial rate gently
            rate = (self.total_blinks / elapsed) * 60.0
            return round(rate, 1)
        # Use rolling 60-second window
        return len(self.blink_timestamps)

    def get_time_since_last_blink(self):
        """Elapsed seconds since the last registered blink."""
        return time.time() - self.last_blink_time


# ==============================================================================
# HUD & UI Rendering Engine
# ==============================================================================
class HUDDrawer:
    """Renders semi-transparent glassmorphic overlays and analytics on OpenCV frames."""
    def __init__(self):
        self.show_mesh = False
        self.show_graph = True
        self.show_help = False

    @staticmethod
    def draw_glass_card(frame, x1, y1, x2, y2, bg_color=COLOR_BG_CARD, alpha=0.72, border_color=COLOR_BORDER):
        """Draws a semi-transparent glassmorphic panel with border."""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        sub_frame = frame[y1:y2, x1:x2]
        overlay = np.full_like(sub_frame, bg_color, dtype=np.uint8)
        frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, sub_frame, 1.0 - alpha, 0)
        if border_color is not None:
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_progress_bar(frame, x, y, width, height, progress, label, color_open=COLOR_ACCENT_GREEN, color_closed=COLOR_ACCENT_RED):
        """Renders an eye openness bar with dynamic color grading."""
        progress = max(0.0, min(1.0, progress))
        # Background slot
        cv2.rectangle(frame, (x, y), (x + width, y + height), (35, 40, 50), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 80, 100), 1)

        # Color interpolation based on openness (1.0 = open green, 0.0 = closed red)
        fill_w = int(width * progress)
        if progress > 0.5:
            bar_color = color_open
        elif progress > 0.25:
            bar_color = COLOR_ACCENT_YELLOW
        else:
            bar_color = color_closed

        if fill_w > 0:
            cv2.rectangle(frame, (x + 1, y + 1), (x + fill_w - 1, y + height - 1), bar_color, -1)

        # Label and percentage text
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, pct_text, (x + width - 36, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

    def draw_sparkline_graph(self, frame, x, y, width, height, history, threshold):
        """Draws rolling eye closure graph with threshold indicator."""
        self.draw_glass_card(frame, x, y, x + width, y + height, bg_color=(16, 18, 26), alpha=0.82)
        cv2.putText(frame, "EYE CLOSURE DYNAMICS", (x + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Threshold guide line
        thresh_y = int(y + height - (threshold * (height - 30)) - 10)
        cv2.line(frame, (x + 10, thresh_y), (x + width - 10, thresh_y), (80, 100, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "BLINK THRESHOLD", (x + width - 110, thresh_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 120, 240), 1, cv2.LINE_AA)

        if len(history) < 2:
            return

        points = []
        n = len(history)
        step = (width - 20) / max(1, (history.maxlen - 1))
        for i, val in enumerate(history):
            px = int(x + 10 + i * step)
            # Invert: val=0 (open) is near bottom, val=1 (closed) near top
            py = int(y + height - 10 - (val * (height - 30)))
            points.append((px, py))

        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i + 1]
            cv2.line(frame, pt1, pt2, COLOR_ACCENT_CYAN, 2, cv2.LINE_AA)

    def draw_landmarks(self, frame, landmarks):
        """Visualizes delicate cyber-contours around eyes and irises."""
        if landmarks is None or len(landmarks) == 0:
            return

        h, w = frame.shape[:2]
        face = landmarks[0]

        # Left Eye Contour
        left_pts = [(int(face[i].x * w), int(face[i].y * h)) for i in LEFT_EYE_CONTOUR]
        if len(left_pts) > 2:
            cv2.polylines(frame, [np.array(left_pts, np.int32)], isClosed=True, color=(255, 230, 0), thickness=1, lineType=cv2.LINE_AA)

        # Right Eye Contour
        right_pts = [(int(face[i].x * w), int(face[i].y * h)) for i in RIGHT_EYE_CONTOUR]
        if len(right_pts) > 2:
            cv2.polylines(frame, [np.array(right_pts, np.int32)], isClosed=True, color=(255, 230, 0), thickness=1, lineType=cv2.LINE_AA)

        # Iris Centers
        for iris_idx in (468, 473):
            if iris_idx < len(face):
                ix = int(face[iris_idx].x * w)
                iy = int(face[iris_idx].y * h)
                cv2.circle(frame, (ix, iy), 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 6, (0, 200, 255), 1, cv2.LINE_AA)

    def draw_hud(self, frame, tracker, left_score, right_score, landmarks, fps, sound_enabled, is_calibrating=False):
        """Composes complete heads-up display overlay."""
        h, w = frame.shape[:2]

        # Draw mesh if enabled
        if self.show_mesh and landmarks:
            self.draw_landmarks(frame, landmarks)

        # ----------------------------------------------------------------------
        # Top Header Bar
        # ----------------------------------------------------------------------
        self.draw_glass_card(frame, 0, 0, w, 50, bg_color=(12, 15, 22), alpha=0.88, border_color=None)
        cv2.putText(frame, "EYE BLINK TRACKER AI", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.72, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # Status Badges
        face_detected = landmarks is not None and len(landmarks) > 0
        status_text = "FACE DETECTED" if face_detected else "NO FACE DETECTED"
        status_color = COLOR_ACCENT_GREEN if face_detected else COLOR_ACCENT_RED
        cv2.circle(frame, (w - 230, 25), 5, status_color, -1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (w - 215, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # FPS counter
        fps_text = f"{int(fps)} FPS"
        cv2.putText(frame, fps_text, (w - 75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # Left Panel: Blink Counter Card
        # ----------------------------------------------------------------------
        card_w = 230
        card_h = 165
        cx1, cy1 = 20, 65
        cx2, cy2 = cx1 + card_w, cy1 + card_h

        # Pulse effect when blink occurs
        if tracker.pulse_frames > 0:
            border_c = COLOR_ACCENT_CYAN
            card_bg = (35, 50, 60)
            glow_thickness = 2
        else:
            border_c = COLOR_BORDER
            card_bg = COLOR_BG_CARD
            glow_thickness = 1

        self.draw_glass_card(frame, cx1, cy1, cx2, cy2, bg_color=card_bg, alpha=0.82, border_color=border_c)
        if glow_thickness > 1:
            cv2.rectangle(frame, (cx1 - 1, cy1 - 1), (cx2 + 1, cy2 + 1), COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)

        cv2.putText(frame, "TOTAL BLINKS", (cx1 + 16, cy1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Big Number
        blinks_str = str(tracker.total_blinks)
        font_scale = 1.6 if tracker.pulse_frames > 0 else 1.5
        count_color = (255, 255, 255) if tracker.pulse_frames > 0 else COLOR_ACCENT_CYAN
        cv2.putText(frame, blinks_str, (cx1 + 16, cy1 + 82), cv2.FONT_HERSHEY_DUPLEX, font_scale, count_color, 2, cv2.LINE_AA)

        # Sub-stats: Winks
        cv2.line(frame, (cx1 + 15, cy1 + 105), (cx2 - 15, cy1 + 105), (50, 60, 80), 1)
        winks_str = f"L-Winks: {tracker.left_winks}    R-Winks: {tracker.right_winks}"
        cv2.putText(frame, winks_str, (cx1 + 16, cy1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Status string: e.g., Blinking / Eyes Open
        state_tag = "BLINKING" if tracker.is_eyes_closed else "EYES OPEN"
        state_c = COLOR_ACCENT_YELLOW if tracker.is_eyes_closed else COLOR_ACCENT_GREEN
        cv2.putText(frame, state_tag, (cx1 + 16, cy1 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.38, state_c, 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # Real-time Eye Gauges (Left & Right Eye Openness)
        # ----------------------------------------------------------------------
        gauge_y = cy2 + 15
        gauge_h = 100
        self.draw_glass_card(frame, cx1, gauge_y, cx2, gauge_y + gauge_h, bg_color=COLOR_BG_CARD, alpha=0.82)
        cv2.putText(frame, "EYE OPENNESS", (cx1 + 16, gauge_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Openness = 1.0 - closure_score
        left_openness = 1.0 - (left_score if left_score is not None else 0.0)
        right_openness = 1.0 - (right_score if right_score is not None else 0.0)

        self.draw_progress_bar(frame, cx1 + 16, gauge_y + 44, 195, 12, left_openness, "Left Eye")
        self.draw_progress_bar(frame, cx1 + 16, gauge_y + 76, 195, 12, right_openness, "Right Eye")

        # ----------------------------------------------------------------------
        # Right Panel: Ergonomics & Health Insights
        # ----------------------------------------------------------------------
        rcard_w = 230
        rcard_h = 165
        rx2 = w - 20
        rx1 = rx2 - rcard_w
        ry1 = 65
        ry2 = ry1 + rcard_h

        self.draw_glass_card(frame, rx1, ry1, rx2, ry2, bg_color=COLOR_BG_CARD, alpha=0.82)
        cv2.putText(frame, "ERGONOMICS & HEALTH", (rx1 + 16, ry1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        bpm = tracker.get_bpm()
        bpm_str = f"{bpm}"
        cv2.putText(frame, bpm_str, (rx1 + 16, ry1 + 68), cv2.FONT_HERSHEY_DUPLEX, 1.2, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "Blinks / Min", (rx1 + 95, ry1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Health assessment indicator
        if bpm >= 14:
            health_text = "Optimal Rate"
            health_c = COLOR_ACCENT_GREEN
        elif bpm >= 9:
            health_text = "Moderate (Rest Eyes)"
            health_c = COLOR_ACCENT_YELLOW
        else:
            health_text = "Low Rate (Fatigue)"
            health_c = COLOR_ACCENT_RED

        cv2.circle(frame, (rx1 + 22, ry1 + 92), 4, health_c, -1, cv2.LINE_AA)
        cv2.putText(frame, health_text, (rx1 + 32, ry1 + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, health_c, 1, cv2.LINE_AA)

        cv2.line(frame, (rx1 + 15, ry1 + 110), (rx2 - 15, ry1 + 110), (50, 60, 80), 1)

        # Time since last blink
        time_since = tracker.get_time_since_last_blink()
        since_c = COLOR_ACCENT_RED if time_since >= STARE_ALERT_SECONDS else COLOR_TEXT_WHITE
        cv2.putText(frame, f"Last Blink: {time_since:.1f}s ago", (rx1 + 16, ry1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, since_c, 1, cv2.LINE_AA)

        # Session Time
        elapsed_sec = int(time.time() - tracker.session_start)
        elapsed_str = f"Session: {elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}"
        cv2.putText(frame, elapsed_str, (rx1 + 16, ry1 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # Dry-Eye / Stare Alert Banner (If user hasn't blinked for > 12s)
        # ----------------------------------------------------------------------
        if face_detected and time_since >= STARE_ALERT_SECONDS:
            # Pulsing warning banner in upper center
            banner_w = 340
            banner_h = 44
            bx1 = (w - banner_w) // 2
            by1 = 65
            bx2 = bx1 + banner_w
            by2 = by1 + banner_h

            pulse = int(abs(math.sin(time.time() * 4.0)) * 60)
            alert_bg = (20, 25, 120 + pulse)
            self.draw_glass_card(frame, bx1, by1, bx2, by2, bg_color=alert_bg, alpha=0.90, border_color=COLOR_ACCENT_RED)
            cv2.putText(frame, "! DRY EYE ALERT: REMEMBER TO BLINK !", (bx1 + 18, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # Live Rolling Sparkline Graph (Bottom Center)
        # ----------------------------------------------------------------------
        if self.show_graph:
            gw = 360
            gh = 95
            gx = (w - gw) // 2
            gy = h - gh - 55
            self.draw_sparkline_graph(frame, gx, gy, gw, gh, tracker.closure_history, tracker.close_threshold)

        # ----------------------------------------------------------------------
        # Calibration Overlay
        # ----------------------------------------------------------------------
        if is_calibrating:
            calib_w = 400
            calib_h = 75
            cx = (w - calib_w) // 2
            cy = h // 2 - 40
            self.draw_glass_card(frame, cx, cy, cx + calib_w, cy + calib_h, bg_color=(40, 30, 20), alpha=0.90, border_color=COLOR_ACCENT_YELLOW)
            cv2.putText(frame, "CALIBRATING EYE SENSITIVITY...", (cx + 20, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ACCENT_YELLOW, 2, cv2.LINE_AA)
            cv2.putText(frame, "Look naturally at the camera for 3 seconds", (cx + 20, cy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # Bottom Controls / Hotkey Help Bar
        # ----------------------------------------------------------------------
        self.draw_glass_card(frame, 0, h - 42, w, h, bg_color=(10, 12, 18), alpha=0.88, border_color=None)
        snd_label = "ON" if sound_enabled else "OFF"
        mesh_label = "ON" if self.show_mesh else "OFF"
        graph_label = "ON" if self.show_graph else "OFF"

        help_bar = f"[R] Reset   [S] Sound: {snd_label}   [M] Mesh: {mesh_label}   [G] Graph: {graph_label}   [C] Calibrate   [H] Help   [Q] Exit"
        cv2.putText(frame, help_bar, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # Help Modal (Toggled with 'H')
        # ----------------------------------------------------------------------
        if self.show_help:
            hw, hh = 440, 240
            hx1, hy1 = (w - hw) // 2, (h - hh) // 2
            self.draw_glass_card(frame, hx1, hy1, hx1 + hw, hy1 + hh, bg_color=(15, 18, 28), alpha=0.94, border_color=COLOR_ACCENT_CYAN)
            cv2.putText(frame, "KEYBOARD SHORTCUTS", (hx1 + 20, hy1 + 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

            lines = [
                ("R", "Reset all counters and session timer"),
                ("S", "Toggle sound feedback on blink / wink"),
                ("M", "Toggle face & eye landmark contours"),
                ("G", "Toggle real-time eyelid dynamics graph"),
                ("C", "Run 3-second sensitivity auto-calibration"),
                ("H", "Show / hide this help panel"),
                ("Q / ESC", "Quit the application")
            ]
            for idx, (key, desc) in enumerate(lines):
                ly = hy1 + 65 + idx * 24
                cv2.putText(frame, f"[{key}]", (hx1 + 25, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.putText(frame, desc, (hx1 + 110, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)


# ==============================================================================
# Calibration Engine
# ==============================================================================
class Calibrator:
    """Samples open-eye and closed-eye values to calibrate optimal thresholds."""
    def __init__(self):
        self.active = False
        self.start_time = 0
        self.duration = 3.0
        self.samples = []

    def start(self):
        self.active = True
        self.start_time = time.time()
        self.samples = []

    def update(self, left_score, right_score):
        if not self.active:
            return None
        if left_score is not None and right_score is not None:
            self.samples.append((left_score + right_score) / 2.0)

        if time.time() - self.start_time >= self.duration:
            self.active = False
            if len(self.samples) > 10:
                baseline_open = float(np.percentile(self.samples, 50))
                new_close_thresh = min(0.65, max(0.38, baseline_open + 0.30))
                new_open_thresh = max(0.15, baseline_open + 0.10)
                print(f"[Calibrator] Calibrated baseline open: {baseline_open:.2f} -> Close Thresh: {new_close_thresh:.2f}, Open Thresh: {new_open_thresh:.2f}")
                return new_close_thresh, new_open_thresh
        return None


# ==============================================================================
# Main Application
# ==============================================================================
def main():
    print("=" * 60)
    print("  EYE BLINK TRACKER AI")
    print("  Powered by MediaPipe FaceLandmarker & OpenCV")
    print("=" * 60)

    # 1. Ensure Model
    model_path = ensure_model_asset(MODEL_FILENAME)

    # 2. Initialize MediaPipe Detector
    print("[Init] Initializing FaceLandmarker neural model...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # 3. Open Camera
    camera_index = 0
    print(f"[Init] Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[Warning] CAP_DSHOW failed, falling back to default backend...")
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_index}. Please verify webcam connection.")
        return

    # Request high resolution (720p or 480p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize Modules
    sound = SoundFeedback(enabled=True)
    tracker = BlinkTracker()
    hud = HUDDrawer()
    calibrator = Calibrator()

    window_name = "Eye Blink Tracker AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS Calculation
    fps_time = time.time()
    fps_count = 0
    fps = 30.0

    print("\n[Running] App is active!")
    print("  Controls: [R] Reset | [S] Sound | [M] Mesh | [G] Graph | [C] Calibrate | [Q] Exit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[Warning] Failed to read frame from webcam.")
                time.sleep(0.01)
                continue

            # Mirror frame for natural selfie feel
            frame = cv2.flip(frame, 1)

            # Compute FPS
            fps_count += 1
            now = time.time()
            if now - fps_time >= 0.5:
                fps = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now

            # MediaPipe inference requires RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)

            left_score = None
            right_score = None
            landmarks = detection_result.face_landmarks

            # Extract blendshapes
            if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
                shapes = {c.category_name: c.score for c in detection_result.face_blendshapes[0]}
                left_score = shapes.get('eyeBlinkLeft', 0.0)
                right_score = shapes.get('eyeBlinkRight', 0.0)

            # Calibration handling
            if calibrator.active:
                new_thresholds = calibrator.update(left_score, right_score)
                if new_thresholds:
                    tracker.close_threshold, tracker.open_threshold = new_thresholds

            # Update Tracker state
            event = tracker.update(left_score, right_score)
            if event == "BLINK":
                sound.play('blink')
            elif event in ("LEFT_WINK", "RIGHT_WINK"):
                sound.play('wink')

            # Render HUD
            hud.draw_hud(
                frame=frame,
                tracker=tracker,
                left_score=left_score,
                right_score=right_score,
                landmarks=landmarks,
                fps=fps,
                sound_enabled=sound.enabled,
                is_calibrating=calibrator.active
            )

            # Show Frame
            cv2.imshow(window_name, frame)

            # Handle Keyboard Inputs
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                print("[Exit] Quitting application...")
                break
            elif key in (ord('r'), ord('R')):
                tracker.reset()
                print("[Action] Counters reset.")
            elif key in (ord('s'), ord('S')):
                state = sound.toggle()
                print(f"[Action] Sound toggled: {'ON' if state else 'OFF'}")
            elif key in (ord('m'), ord('M')):
                hud.show_mesh = not hud.show_mesh
                print(f"[Action] Mesh overlay: {'ON' if hud.show_mesh else 'OFF'}")
            elif key in (ord('g'), ord('G')):
                hud.show_graph = not hud.show_graph
                print(f"[Action] Graph: {'ON' if hud.show_graph else 'OFF'}")
            elif key in (ord('h'), ord('H')):
                hud.show_help = not hud.show_help
            elif key in (ord('c'), ord('C')):
                calibrator.start()
                print("[Action] Calibration started (3s)...")

    except KeyboardInterrupt:
        print("\n[Interrupted] Terminating...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Shutdown] Clean exit.")


if __name__ == "__main__":
    main()
