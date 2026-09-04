#!/usr/bin/env python3
"""
Ocular AI - Windows Desktop Application
--------------------------------------
A modern, dark-mode Windows desktop GUI for real-time Eye Blink and Yawn tracking,
health analytics, and session history management powered by MediaPipe, CustomTkinter,
and SQLite.
"""

import os
import sys
import time
import math
import queue
import threading
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import messagebox

# MediaPipe & Local Modules
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import blink_counter
import storage

# Set CustomTkinter Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Theme Palette
COLOR_BG_DARK = "#0B0E14"
COLOR_CARD_BG = "#151923"
COLOR_CARD_BORDER = "#232A3B"
COLOR_CARD_ACCENT = "#1E2536"
COLOR_CYAN = "#00E5FF"
COLOR_GREEN = "#10B981"
COLOR_YELLOW = "#FBBF24"
COLOR_PURPLE = "#C084FC"
COLOR_RED = "#EF4444"
COLOR_TEXT_DIM = "#94A3B8"
COLOR_TEXT_WHITE = "#F8FAFC"


class OcularDesktopApp(ctk.CTk):
    """Main CustomTkinter Window for Ocular AI Desktop Application."""
    def __init__(self):
        super().__init__()

        self.title("Ocular AI — Blink & Yawn Ergonomics Tracker")
        self.geometry("1200x720")
        self.minsize(1020, 640)
        self.configure(fg_color=COLOR_BG_DARK)

        # Database & Storage
        self.db = storage.StorageManager()

        # Detection & Camera State
        self.tracker = blink_counter.BlinkTracker()
        self.sound = blink_counter.SoundFeedback(enabled=True)
        self.is_camera_running = False
        self.cap = None
        self.detector = None
        self.camera_index = 0
        self.show_mesh = False
        self.mirror_feed = True
        self.latest_frame = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        # Session metrics
        self.session_start_time = time.time()
        self.fps = 0.0
        self.fps_counter = 0
        self.fps_timer = time.time()

        # Settings
        self.stare_alert_sec = 12.0
        self.auto_save = True

        # Initialize Neural Detector
        self._init_detector()

        # Build UI
        self._build_ui()

        # Start Camera Thread
        self.start_camera()

        # Schedule Periodic UI Update Loop
        self.after(30, self._process_video_frames)
        self.after(500, self._update_metrics_cards)

        # Handle Close Event
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_detector(self):
        """Initializes the MediaPipe FaceLandmarker task."""
        model_path = blink_counter.ensure_model_asset()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _build_ui(self):
        """Constructs the desktop application layout."""
        # ======================================================================
        # Header Bar
        # ======================================================================
        self.header_frame = ctk.CTkFrame(self, fg_color="#141824", height=60, corner_radius=0)
        self.header_frame.pack(fill="x", side="top")
        self.header_frame.pack_propagate(False)

        # Logo & App Title
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="👁️ OCULAR AI",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLOR_TEXT_WHITE
        )
        self.title_label.pack(side="left", padx=20)

        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="|  Intelligent Blink & Yawn Monitor",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_TEXT_DIM
        )
        self.subtitle_label.pack(side="left", padx=5)

        # Camera Status & FPS indicator
        self.status_badge = ctk.CTkLabel(
            self.header_frame,
            text="🟢 CAMERA ACTIVE",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLOR_GREEN
        )
        self.status_badge.pack(side="right", padx=20)

        self.fps_badge = ctk.CTkLabel(
            self.header_frame,
            text="30 FPS",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_CYAN
        )
        self.fps_badge.pack(side="right", padx=15)

        # ======================================================================
        # Tab View Navigation
        # ======================================================================
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=COLOR_BG_DARK,
            segmented_button_fg_color="#1E2333",
            segmented_button_selected_color="#2563EB",
            segmented_button_selected_hover_color="#1D4ED8",
            segmented_button_unselected_color="#1E2333",
            segmented_button_unselected_hover_color="#2B3247",
            text_color=COLOR_TEXT_WHITE
        )
        self.tabview.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # Define Tabs
        self.tab_monitor = self.tabview.add("  🖥️  Live Monitor  ")
        self.tab_history = self.tabview.add("  📊  History & Analytics  ")
        self.tab_settings = self.tabview.add("  ⚙️  Settings  ")

        # Build each tab's contents
        self._build_monitor_tab()
        self._build_history_tab()
        self._build_settings_tab()

    # --------------------------------------------------------------------------
    # Tab 1: Live Monitor
    # --------------------------------------------------------------------------
    def _build_monitor_tab(self):
        parent = self.tab_monitor

        # Split into Left (Flexible Camera Stream) and Right (Dedicated 400px Metrics Dashboard)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=0, minsize=400)
        parent.rowconfigure(0, weight=1)

        # Left: Video Frame Container
        self.video_container = ctk.CTkFrame(
            parent,
            fg_color=COLOR_CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=(6, 8), pady=10)

        # Video Bottom Controls Bar (packed first at bottom so it is NEVER clipped)
        self.video_controls = ctk.CTkFrame(self.video_container, fg_color="transparent", height=48)
        self.video_controls.pack(side="bottom", fill="x", padx=14, pady=(6, 12))
        self.video_controls.pack_propagate(False)

        self.btn_cam_toggle = ctk.CTkButton(
            self.video_controls,
            text="⏸️ Pause Camera",
            width=130,
            height=36,
            corner_radius=8,
            command=self.toggle_camera,
            fg_color="#1E2536",
            hover_color="#2B354C",
            border_width=1,
            border_color="#2D374D"
        )
        self.btn_cam_toggle.pack(side="left", padx=(0, 8))

        self.btn_mesh_toggle = ctk.CTkButton(
            self.video_controls,
            text="🕸️ Mesh: OFF",
            width=115,
            height=36,
            corner_radius=8,
            command=self.toggle_mesh,
            fg_color="#1E2536",
            hover_color="#2B354C",
            border_width=1,
            border_color="#2D374D"
        )
        self.btn_mesh_toggle.pack(side="left", padx=4)

        self.btn_sound_toggle = ctk.CTkButton(
            self.video_controls,
            text="🔊 Sound: ON",
            width=115,
            height=36,
            corner_radius=8,
            command=self.toggle_sound,
            fg_color="#1E2536",
            hover_color="#2B354C",
            border_width=1,
            border_color="#2D374D"
        )
        self.btn_sound_toggle.pack(side="left", padx=4)

        self.btn_reset = ctk.CTkButton(
            self.video_controls,
            text="🔄 Reset Counter",
            width=130,
            height=36,
            corner_radius=8,
            command=self.reset_session,
            fg_color="#DC2626",
            hover_color="#B91C1C"
        )
        self.btn_reset.pack(side="right", padx=(8, 0))

        # Video Viewport Container (takes remaining vertical space, prevents expansion loops)
        self.video_viewport = ctk.CTkFrame(self.video_container, fg_color="#07090E", corner_radius=8)
        self.video_viewport.pack(side="top", fill="both", expand=True, padx=14, pady=(12, 6))
        self.video_viewport.pack_propagate(False)

        # Video Canvas / Centered Label
        self.video_label = ctk.CTkLabel(
            self.video_viewport,
            text="Connecting Camera...",
            font=ctk.CTkFont(size=13),
            text_color=COLOR_TEXT_DIM,
            fg_color="transparent"
        )
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # Right: Real-time Analytics Cards (Fixed 400px width, zero truncation)
        self.dash_frame = ctk.CTkScrollableFrame(parent, width=400, fg_color="transparent")
        self.dash_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 8), pady=10)

        # 1. Total Blinks Card
        self.card_blinks = ctk.CTkFrame(
            self.dash_frame,
            fg_color=COLOR_CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.card_blinks.pack(fill="x", pady=(0, 10))

        blinks_title = ctk.CTkLabel(
            self.card_blinks,
            text="👁️ TOTAL BLINKS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_CYAN
        )
        blinks_title.pack(anchor="w", padx=16, pady=(12, 2))

        self.lbl_blink_count = ctk.CTkLabel(
            self.card_blinks,
            text="0",
            font=ctk.CTkFont(family="Segoe UI", size=42, weight="bold"),
            text_color=COLOR_CYAN
        )
        self.lbl_blink_count.pack(anchor="w", padx=16, pady=(0, 2))

        self.lbl_winks = ctk.CTkLabel(
            self.card_blinks,
            text="Left Winks: 0   •   Right Winks: 0",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_DIM
        )
        self.lbl_winks.pack(anchor="w", padx=16, pady=(0, 12))

        # 2. Total Yawns Card
        self.card_yawns = ctk.CTkFrame(
            self.dash_frame,
            fg_color=COLOR_CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.card_yawns.pack(fill="x", pady=(0, 10))

        yawns_title = ctk.CTkLabel(
            self.card_yawns,
            text="😮 TOTAL YAWNS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_PURPLE
        )
        yawns_title.pack(anchor="w", padx=16, pady=(12, 2))

        self.lbl_yawn_count = ctk.CTkLabel(
            self.card_yawns,
            text="0",
            font=ctk.CTkFont(family="Segoe UI", size=42, weight="bold"),
            text_color=COLOR_PURPLE
        )
        self.lbl_yawn_count.pack(anchor="w", padx=16, pady=(0, 2))

        self.lbl_yawn_status = ctk.CTkLabel(
            self.card_yawns,
            text="Mouth: Normal (Threshold: 58%)",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_GREEN
        )
        self.lbl_yawn_status.pack(anchor="w", padx=16, pady=(0, 12))

        # 3. Health & Ergonomics Card (BPM + Drowsiness)
        self.card_health = ctk.CTkFrame(
            self.dash_frame,
            fg_color=COLOR_CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.card_health.pack(fill="x", pady=(0, 10))

        health_title = ctk.CTkLabel(
            self.card_health,
            text="⚡ OCULAR HEALTH & RATE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#38BDF8"
        )
        health_title.pack(anchor="w", padx=16, pady=(12, 2))

        bpm_box = ctk.CTkFrame(self.card_health, fg_color="transparent")
        bpm_box.pack(anchor="w", padx=16, pady=(2, 2))

        self.lbl_bpm = ctk.CTkLabel(
            bpm_box,
            text="0.0",
            font=ctk.CTkFont(family="Segoe UI", size=32, weight="bold"),
            text_color=COLOR_TEXT_WHITE
        )
        self.lbl_bpm.pack(side="left")

        lbl_bpm_unit = ctk.CTkLabel(
            bpm_box,
            text="  Blinks / Min",
            font=ctk.CTkFont(size=13),
            text_color=COLOR_TEXT_DIM
        )
        lbl_bpm_unit.pack(side="left", padx=4)

        self.lbl_alertness = ctk.CTkLabel(
            self.card_health,
            text="🟢 Normal & Attentive",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_GREEN
        )
        self.lbl_alertness.pack(anchor="w", padx=16, pady=(2, 4))

        self.lbl_session_time = ctk.CTkLabel(
            self.card_health,
            text="Session: 00:00   •   Last Blink: 0.0s ago",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_DIM
        )
        self.lbl_session_time.pack(anchor="w", padx=16, pady=(0, 12))

        # 4. Facial Openness Real-time Gauges
        self.card_gauges = ctk.CTkFrame(
            self.dash_frame,
            fg_color=COLOR_CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.card_gauges.pack(fill="x", pady=(0, 10))

        gauges_title = ctk.CTkLabel(
            self.card_gauges,
            text="📊 REAL-TIME FACIAL DYNAMICS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_WHITE
        )
        gauges_title.pack(anchor="w", padx=16, pady=(12, 10))

        # Left Eye Bar
        row_le = ctk.CTkFrame(self.card_gauges, fg_color="transparent")
        row_le.pack(fill="x", padx=16)
        ctk.CTkLabel(row_le, text="Left Eye", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_DIM).pack(side="left")
        self.lbl_left_eye = ctk.CTkLabel(row_le, text="100% Open", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_WHITE)
        self.lbl_left_eye.pack(side="right")

        self.bar_left_eye = ctk.CTkProgressBar(self.card_gauges, progress_color=COLOR_GREEN, fg_color="#222938", height=8)
        self.bar_left_eye.pack(fill="x", padx=16, pady=(2, 8))
        self.bar_left_eye.set(1.0)

        # Right Eye Bar
        row_re = ctk.CTkFrame(self.card_gauges, fg_color="transparent")
        row_re.pack(fill="x", padx=16)
        ctk.CTkLabel(row_re, text="Right Eye", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_DIM).pack(side="left")
        self.lbl_right_eye = ctk.CTkLabel(row_re, text="100% Open", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_WHITE)
        self.lbl_right_eye.pack(side="right")

        self.bar_right_eye = ctk.CTkProgressBar(self.card_gauges, progress_color=COLOR_GREEN, fg_color="#222938", height=8)
        self.bar_right_eye.pack(fill="x", padx=16, pady=(2, 8))
        self.bar_right_eye.set(1.0)

        # Mouth / Jaw Bar
        row_m = ctk.CTkFrame(self.card_gauges, fg_color="transparent")
        row_m.pack(fill="x", padx=16)
        ctk.CTkLabel(row_m, text="Mouth Openness", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_DIM).pack(side="left")
        self.lbl_mouth = ctk.CTkLabel(row_m, text="Resting (0%)", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_WHITE)
        self.lbl_mouth.pack(side="right")

        self.bar_mouth = ctk.CTkProgressBar(self.card_gauges, progress_color=COLOR_PURPLE, fg_color="#222938", height=8)
        self.bar_mouth.pack(fill="x", padx=16, pady=(2, 14))
        self.bar_mouth.set(0.0)

    # --------------------------------------------------------------------------
    # Tab 2: History & Analytics
    # --------------------------------------------------------------------------
    def _build_history_tab(self):
        parent = self.tab_history

        # Lifetime Stats Banner
        self.lifetime_frame = ctk.CTkFrame(parent, fg_color=COLOR_CARD_BG, corner_radius=12)
        self.lifetime_frame.pack(fill="x", padx=10, pady=10)

        self.lifetime_frame.columnconfigure((0, 1, 2, 3), weight=1)

        # Lifetime Blinks
        f1 = ctk.CTkFrame(self.lifetime_frame, fg_color="transparent")
        f1.grid(row=0, column=0, pady=12)
        ctk.CTkLabel(f1, text="LIFETIME BLINKS", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_DIM).pack()
        self.stat_life_blinks = ctk.CTkLabel(f1, text="0", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), text_color=COLOR_CYAN)
        self.stat_life_blinks.pack()

        # Lifetime Yawns
        f2 = ctk.CTkFrame(self.lifetime_frame, fg_color="transparent")
        f2.grid(row=0, column=1, pady=12)
        ctk.CTkLabel(f2, text="LIFETIME YAWNS", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_DIM).pack()
        self.stat_life_yawns = ctk.CTkLabel(f2, text="0", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), text_color=COLOR_PURPLE)
        self.stat_life_yawns.pack()

        # Total Sessions
        f3 = ctk.CTkFrame(self.lifetime_frame, fg_color="transparent")
        f3.grid(row=0, column=2, pady=12)
        ctk.CTkLabel(f3, text="TOTAL SESSIONS", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_DIM).pack()
        self.stat_life_sessions = ctk.CTkLabel(f3, text="0", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), text_color=COLOR_GREEN)
        self.stat_life_sessions.pack()

        # Hours Tracked
        f4 = ctk.CTkFrame(self.lifetime_frame, fg_color="transparent")
        f4.grid(row=0, column=3, pady=12)
        ctk.CTkLabel(f4, text="TOTAL HOURS", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_DIM).pack()
        self.stat_life_hours = ctk.CTkLabel(f4, text="0.0h", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), text_color=COLOR_YELLOW)
        self.stat_life_hours.pack()

        # Toolbar
        self.history_toolbar = ctk.CTkFrame(parent, fg_color="transparent")
        self.history_toolbar.pack(fill="x", padx=10, pady=(0, 10))

        self.btn_refresh_history = ctk.CTkButton(
            self.history_toolbar,
            text="🔄 Refresh",
            width=110,
            height=36,
            corner_radius=8,
            command=self.load_history_table,
            fg_color="#2563EB",
            hover_color="#1D4ED8"
        )
        self.btn_refresh_history.pack(side="left", padx=5)

        self.btn_export_csv = ctk.CTkButton(
            self.history_toolbar,
            text="📥 Export to CSV",
            width=130,
            height=36,
            corner_radius=8,
            command=self.export_csv,
            fg_color="#059669",
            hover_color="#047857"
        )
        self.btn_export_csv.pack(side="left", padx=5)

        # In-toolbar status message (shows green toasts for clear/export)
        self.history_status_lbl = ctk.CTkLabel(
            self.history_toolbar,
            text="",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.history_status_lbl.pack(side="left", padx=15)

        self.btn_clear_history = ctk.CTkButton(
            self.history_toolbar,
            text="🗑️ Clear History",
            width=130,
            height=36,
            corner_radius=8,
            command=self.clear_history,
            fg_color="#DC2626",
            hover_color="#B91C1C"
        )
        self.btn_clear_history.pack(side="right", padx=5)

        # Table Container
        self.table_scroll = ctk.CTkScrollableFrame(
            parent,
            fg_color=COLOR_CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.table_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Initial history load
        self.load_history_table()

    def _show_history_status(self, text, color):
        """Displays a temporary notification banner on the history toolbar."""
        self.history_status_lbl.configure(text=text, text_color=color)
        self.after(4500, lambda: self.history_status_lbl.configure(text=""))

    def load_history_table(self):
        """Loads sessions and lifetime stats from SQLite into the GUI table."""
        # Update lifetime stats
        stats = self.db.get_lifetime_stats()
        self.stat_life_blinks.configure(text=f"{stats['lifetime_blinks']:,}")
        self.stat_life_yawns.configure(text=f"{stats['lifetime_yawns']:,}")
        self.stat_life_sessions.configure(text=f"{stats['total_sessions']:,}")
        self.stat_life_hours.configure(text=f"{stats['lifetime_hours']}h")

        # Clear existing rows in table
        for widget in self.table_scroll.winfo_children():
            widget.destroy()

        # Table Header
        headers = ["ID", "Start Time", "Duration", "Blinks", "Yawns", "Winks", "Avg BPM", "Fatigue Assessment"]
        col_weights = [1, 3, 2, 2, 2, 2, 2, 3]

        header_frame = ctk.CTkFrame(self.table_scroll, fg_color="#1E2536", corner_radius=6)
        header_frame.pack(fill="x", pady=(0, 4))
        for idx, (h, w) in enumerate(zip(headers, col_weights)):
            header_frame.columnconfigure(idx, weight=w)
            lbl = ctk.CTkLabel(header_frame, text=h, font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_WHITE)
            lbl.grid(row=0, column=idx, padx=8, pady=8, sticky="w")

        # Fetch recent sessions
        sessions = self.db.get_recent_sessions(limit=50)
        if not sessions:
            empty_box = ctk.CTkFrame(self.table_scroll, fg_color="transparent")
            empty_box.pack(pady=50)
            ctk.CTkLabel(empty_box, text="📭", font=ctk.CTkFont(size=38)).pack(pady=(0, 8))
            ctk.CTkLabel(empty_box, text="No Tracking Sessions Recorded Yet", font=ctk.CTkFont(size=15, weight="bold"), text_color=COLOR_TEXT_WHITE).pack()
            ctk.CTkLabel(empty_box, text="Start tracking blinks or yawns in the Live Monitor to generate history logs.", font=ctk.CTkFont(size=12), text_color=COLOR_TEXT_DIM).pack(pady=(4, 0))
            return

        for r_idx, s in enumerate(sessions):
            bg = "#1A1F2C" if r_idx % 2 == 0 else "#141722"
            row_frame = ctk.CTkFrame(self.table_scroll, fg_color=bg, corner_radius=4)
            row_frame.pack(fill="x", pady=2)

            dur_sec = s["duration_seconds"]
            dur_str = f"{dur_sec // 60}m {dur_sec % 60}s" if dur_sec >= 60 else f"{dur_sec}s"

            items = [
                f"#{s['id']}",
                s["start_time"],
                dur_str,
                f"{s['total_blinks']} blinks",
                f"{s['total_yawns']} yawns",
                f"{s['total_winks']} winks",
                f"{s['avg_bpm']:.1f}",
                s["fatigue_status"]
            ]

            status_color = COLOR_GREEN if "ALERT" in s["fatigue_status"] else (COLOR_YELLOW if "TIRED" in s["fatigue_status"] else COLOR_RED)

            for c_idx, (val, w) in enumerate(zip(items, col_weights)):
                row_frame.columnconfigure(c_idx, weight=w)
                col_color = status_color if c_idx == 7 else COLOR_TEXT_WHITE
                lbl = ctk.CTkLabel(row_frame, text=val, font=ctk.CTkFont(size=12), text_color=col_color)
                lbl.grid(row=0, column=c_idx, padx=8, pady=6, sticky="w")

    def export_csv(self):
        """Export session database to CSV file."""
        try:
            csv_path = self.db.export_to_csv()
            filename = os.path.basename(csv_path)
            print(f"[Export] Saved session history to CSV: {csv_path}")
            self._show_history_status(f"✓ Exported to {filename}", COLOR_GREEN)
        except Exception as e:
            print(f"[Export] Error exporting to CSV: {e}")
            self._show_history_status(f"⚠️ Export error: {e}", COLOR_RED)

    def clear_history(self):
        """Prompts user and purges all sessions from the database and active session."""
        confirm = messagebox.askyesno(
            "Clear Tracking History",
            "Are you sure you want to permanently delete all tracking history and reset the current session counters?\n\nThis action cannot be undone.",
            parent=self
        )
        if not confirm:
            return

        try:
            self.db.clear_history()
            self.tracker.reset()
            self.session_start_time = time.time()
            self._update_metrics_cards()
            self.load_history_table()
            self._show_history_status("✓ All tracking history & current session cleared!", COLOR_GREEN)
            print("[Storage] All tracking history and active session data cleared.")
        except Exception as e:
            print(f"[Storage] Error clearing history: {e}")
            self._show_history_status(f"⚠️ Error clearing history: {e}", COLOR_RED)

    # --------------------------------------------------------------------------
    # Tab 3: Settings
    # --------------------------------------------------------------------------
    def _build_settings_tab(self):
        parent = self.tab_settings

        settings_scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        settings_scroll.pack(fill="both", expand=True, padx=20, pady=15)

        # 1. Camera Settings
        card_cam = ctk.CTkFrame(settings_scroll, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_CARD_BORDER)
        card_cam.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(card_cam, text="CAMERA HARDWARE", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_WHITE).pack(anchor="w", padx=20, pady=(15, 8))

        row_cam = ctk.CTkFrame(card_cam, fg_color="transparent")
        row_cam.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(row_cam, text="Camera Device Index:", font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_DIM).pack(side="left")
        self.opt_cam = ctk.CTkOptionMenu(
            row_cam,
            values=["Camera 0 (Default)", "Camera 1", "Camera 2"],
            command=self.change_camera_device,
            width=180
        )
        self.opt_cam.pack(side="right")

        # 2. Yawn Detection Tuning
        card_yawn = ctk.CTkFrame(settings_scroll, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_CARD_BORDER)
        card_yawn.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(card_yawn, text="YAWN DETECTION SENSITIVITY", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_WHITE).pack(anchor="w", padx=20, pady=(15, 8))

        # Yawn Threshold Slider
        self.lbl_slider_yawn_th = ctk.CTkLabel(card_yawn, text=f"Mouth Open Threshold: {int(self.tracker.yawn_threshold * 100)}%", font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_DIM)
        self.lbl_slider_yawn_th.pack(anchor="w", padx=20, pady=(4, 0))

        self.slider_yawn_th = ctk.CTkSlider(
            card_yawn,
            from_=0.35,
            to=0.75,
            number_of_steps=40,
            command=self.on_change_yawn_threshold
        )
        self.slider_yawn_th.set(self.tracker.yawn_threshold)
        self.slider_yawn_th.pack(fill="x", padx=20, pady=(2, 12))

        # Yawn Hold Duration Slider
        self.lbl_slider_yawn_hold = ctk.CTkLabel(card_yawn, text=f"Required Continuous Hold: {self.tracker.min_yawn_hold_sec:.1f}s", font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_DIM)
        self.lbl_slider_yawn_hold.pack(anchor="w", padx=20, pady=(4, 0))

        self.slider_yawn_hold = ctk.CTkSlider(
            card_yawn,
            from_=0.6,
            to=2.2,
            number_of_steps=16,
            command=self.on_change_yawn_hold
        )
        self.slider_yawn_hold.set(self.tracker.min_yawn_hold_sec)
        self.slider_yawn_hold.pack(fill="x", padx=20, pady=(2, 15))

        # 3. Blink Detection Tuning
        card_blink = ctk.CTkFrame(settings_scroll, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_CARD_BORDER)
        card_blink.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(card_blink, text="BLINK DETECTION SENSITIVITY", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_WHITE).pack(anchor="w", padx=20, pady=(15, 8))

        self.lbl_slider_blink_th = ctk.CTkLabel(card_blink, text=f"Eye Closure Trigger: {int(self.tracker.close_threshold * 100)}%", font=ctk.CTkFont(size=13), text_color=COLOR_TEXT_DIM)
        self.lbl_slider_blink_th.pack(anchor="w", padx=20, pady=(4, 0))

        self.slider_blink_th = ctk.CTkSlider(
            card_blink,
            from_=0.30,
            to=0.65,
            number_of_steps=35,
            command=self.on_change_blink_threshold
        )
        self.slider_blink_th.set(self.tracker.close_threshold)
        self.slider_blink_th.pack(fill="x", padx=20, pady=(2, 15))

        # 4. Storage & Audio Preferences
        card_audio = ctk.CTkFrame(settings_scroll, fg_color=COLOR_CARD_BG, corner_radius=12, border_width=1, border_color=COLOR_CARD_BORDER)
        card_audio.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(card_audio, text="PREFERENCES & AUDIO", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_WHITE).pack(anchor="w", padx=20, pady=(15, 8))

        self.switch_audio = ctk.CTkSwitch(
            card_audio,
            text="Play subtle audio cues on blinks and yawns",
            command=self.toggle_sound,
            font=ctk.CTkFont(size=13)
        )
        self.switch_audio.select()
        self.switch_audio.pack(anchor="w", padx=20, pady=(4, 8))

        # Sound test buttons
        row_test = ctk.CTkFrame(card_audio, fg_color="transparent")
        row_test.pack(anchor="w", padx=20, pady=(0, 12))

        ctk.CTkButton(
            row_test,
            text="🔔 Test Blink Sound",
            width=135,
            height=30,
            corner_radius=6,
            fg_color="#1E2536",
            hover_color="#2B354C",
            command=lambda: self.sound.play('blink')
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            row_test,
            text="🔔 Test Yawn Sound",
            width=135,
            height=30,
            corner_radius=6,
            fg_color="#1E2536",
            hover_color="#2B354C",
            command=lambda: self.sound.play('yawn')
        ).pack(side="left")

        self.switch_autosave = ctk.CTkSwitch(
            card_audio,
            text="Automatically save tracking sessions upon closing",
            font=ctk.CTkFont(size=13)
        )
        self.switch_autosave.select()
        self.switch_autosave.pack(anchor="w", padx=20, pady=(0, 15))

    def on_change_yawn_threshold(self, val):
        self.tracker.yawn_threshold = round(val, 2)
        self.lbl_slider_yawn_th.configure(text=f"Mouth Open Threshold: {int(self.tracker.yawn_threshold * 100)}%")

    def on_change_yawn_hold(self, val):
        self.tracker.min_yawn_hold_sec = round(val, 1)
        self.lbl_slider_yawn_hold.configure(text=f"Required Continuous Hold: {self.tracker.min_yawn_hold_sec:.1f}s")

    def on_change_blink_threshold(self, val):
        self.tracker.close_threshold = round(val, 2)
        self.lbl_slider_blink_th.configure(text=f"Eye Closure Trigger: {int(self.tracker.close_threshold * 100)}%")

    def change_camera_device(self, choice):
        """Switches webcam index."""
        idx = int(choice.split()[1].replace("(Default)", ""))
        if idx != self.camera_index:
            self.camera_index = idx
            self.stop_camera()
            self.start_camera()

    # ==========================================================================
    # Camera Capture & Processing Engine (Threaded)
    # ==========================================================================
    def start_camera(self):
        """Starts background webcam capture worker thread."""
        if self.is_camera_running:
            return

        self.stop_event.clear()
        self.is_camera_running = True
        self.cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.cam_thread.start()
        self.status_badge.configure(text="🟢 CAMERA ACTIVE", text_color=COLOR_GREEN)
        self.btn_cam_toggle.configure(text="⏸️ Pause Camera", fg_color="#1E2536")

    def stop_camera(self):
        """Stops the camera capture thread."""
        self.is_camera_running = False
        self.stop_event.set()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_badge.configure(text="🔴 CAMERA PAUSED", text_color=COLOR_RED)
        self.btn_cam_toggle.configure(text="▶️ Resume Camera", fg_color="#2563EB")

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def toggle_mesh(self):
        self.show_mesh = not self.show_mesh
        self.btn_mesh_toggle.configure(
            text=f"🕸️ Mesh: {'ON' if self.show_mesh else 'OFF'}",
            fg_color="#059669" if self.show_mesh else "#1E2536"
        )

    def toggle_sound(self):
        state = self.sound.toggle()
        self.btn_sound_toggle.configure(
            text=f"🔊 Sound: {'ON' if state else 'OFF'}",
            fg_color="#059669" if state else "#1E2536"
        )
        if hasattr(self, 'switch_audio'):
            if state:
                self.switch_audio.select()
            else:
                self.switch_audio.deselect()

    def reset_session(self):
        """Ends current session, saves to database if active, and resets live counters."""
        if self.tracker.total_blinks > 0 or self.tracker.total_yawns > 0:
            self.save_current_session_to_db()
        self.tracker.reset()
        self.session_start_time = time.time()
        self._update_metrics_cards()
        self.load_history_table()

    def _camera_worker(self):
        """Dedicated background thread for OpenCV capture and MediaPipe inference."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[Error] Could not open webcam index {self.camera_index}")
            self.is_camera_running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                if self.mirror_feed:
                    frame = cv2.flip(frame, 1)

                # FPS calculation
                self.fps_counter += 1
                now = time.time()
                if now - self.fps_timer >= 0.5:
                    self.fps = self.fps_counter / (now - self.fps_timer)
                    self.fps_counter = 0
                    self.fps_timer = now

                # MediaPipe inference requires RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)

                left_score = None
                right_score = None
                raw_jaw_score = None
                landmarks = detection_result.face_landmarks

                if detection_result.face_blendshapes and len(detection_result.face_blendshapes) > 0:
                    shapes = {c.category_name: c.score for c in detection_result.face_blendshapes[0]}
                    left_score = shapes.get('eyeBlinkLeft', 0.0)
                    right_score = shapes.get('eyeBlinkRight', 0.0)
                    raw_jaw_score = shapes.get('jawOpen', 0.0)

                # Calculate ensemble mouth score
                mouth_score, mar = blink_counter.compute_mouth_metrics(landmarks, raw_jaw_score)

                # Update Tracker
                event = self.tracker.update(left_score, right_score, mouth_score)
                if event == "BLINK":
                    self.sound.play('blink')
                elif event == "YAWN":
                    self.sound.play('yawn')
                    print(f"[Event] Yawn #{self.tracker.total_yawns} detected! (Mouth: {int(mouth_score * 100)}%)")
                elif event in ("LEFT_WINK", "RIGHT_WINK"):
                    self.sound.play('wink')

                # Render Landmarks overlay on frame if enabled
                if self.show_mesh and landmarks and len(landmarks) > 0:
                    self._draw_mesh_on_frame(frame, landmarks[0])

                # Push latest processed frame & scores
                meta = {
                    "frame": frame,
                    "left_score": left_score,
                    "right_score": right_score,
                    "mouth_score": mouth_score,
                    "has_face": landmarks is not None and len(landmarks) > 0
                }

                try:
                    self.frame_queue.put_nowait(meta)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(meta)
                    except Exception:
                        pass
            except Exception:
                if self.stop_event.is_set():
                    break
                time.sleep(0.01)

        if self.cap is not None:
            self.cap.release()

    def _draw_mesh_on_frame(self, frame, face):
        """Draws glowing eye and mouth contours directly on video frame."""
        h, w = frame.shape[:2]

        # Eyes (Gold/Cyan)
        for contour in (blink_counter.LEFT_EYE_CONTOUR, blink_counter.RIGHT_EYE_CONTOUR):
            pts = [(int(face[i].x * w), int(face[i].y * h)) for i in contour if i < len(face)]
            if len(pts) > 2:
                cv2.polylines(frame, [np.array(pts, np.int32)], isClosed=True, color=(255, 230, 0), thickness=1, lineType=cv2.LINE_AA)

        # Iris Centers
        for iris_idx in (468, 473):
            if iris_idx < len(face):
                ix, iy = int(face[iris_idx].x * w), int(face[iris_idx].y * h)
                cv2.circle(frame, (ix, iy), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # Lips (Neon Magenta)
        for lip in (blink_counter.LIPS_OUTER, blink_counter.LIPS_INNER):
            pts = [(int(face[i].x * w), int(face[i].y * h)) for i in lip if i < len(face)]
            if len(pts) > 2:
                cv2.polylines(frame, [np.array(pts, np.int32)], isClosed=True, color=(220, 110, 255), thickness=1, lineType=cv2.LINE_AA)

    # ==========================================================================
    # Main Thread UI Update Loop
    # ==========================================================================
    def _process_video_frames(self):
        """Processes latest frame and updates CustomTkinter video label."""
        try:
            meta = self.frame_queue.get_nowait()
            frame = meta["frame"]
            left_score = meta["left_score"]
            right_score = meta["right_score"]
            mouth_score = meta["mouth_score"]
            has_face = meta["has_face"]

            # Resize frame strictly within viewport bounds to prevent expansion loops
            vp_w = self.video_viewport.winfo_width()
            vp_h = self.video_viewport.winfo_height()
            if vp_w <= 30 or vp_h <= 30:
                vp_w = 640
                vp_h = 360

            h, w = frame.shape[:2]
            aspect = w / h
            target_w = vp_w
            target_h = int(target_w / aspect)
            if target_h > vp_h:
                target_h = vp_h
                target_w = int(target_h * aspect)

            target_w = max(10, target_w)
            target_h = max(10, target_h)

            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

            # Live facial openness progress bars & labels
            left_open = 1.0 - (left_score if left_score is not None else 0.0)
            right_open = 1.0 - (right_score if right_score is not None else 0.0)
            mouth_val = mouth_score if mouth_score is not None else 0.0

            self.bar_left_eye.set(left_open)
            self.lbl_left_eye.configure(text=f"{int(left_open * 100)}% Open")

            self.bar_right_eye.set(right_open)
            self.lbl_right_eye.configure(text=f"{int(right_open * 100)}% Open")

            self.bar_mouth.set(mouth_val)
            if not self.tracker.yawn_enabled:
                self.lbl_mouth.configure(text="Disabled", text_color=COLOR_TEXT_DIM)
            elif mouth_val >= self.tracker.yawn_threshold:
                self.lbl_mouth.configure(text=f"YAWN ZONE ({int(mouth_val * 100)}%)", text_color=COLOR_PURPLE)
            elif mouth_val >= 0.20:
                self.lbl_mouth.configure(text=f"Speaking ({int(mouth_val * 100)}%)", text_color=COLOR_YELLOW)
            else:
                self.lbl_mouth.configure(text=f"Resting ({int(mouth_val * 100)}%)", text_color=COLOR_CYAN)

        except queue.Empty:
            pass

        # Re-schedule at ~30 FPS
        self.after(30, self._process_video_frames)

    def _update_metrics_cards(self):
        """Updates numeric dashboard cards and timers."""
        # Blinks Card
        self.lbl_blink_count.configure(text=f"{self.tracker.total_blinks:,}")
        self.lbl_winks.configure(text=f"Left Winks: {self.tracker.left_winks}   •   Right Winks: {self.tracker.right_winks}")

        # Yawns Card
        self.lbl_yawn_count.configure(text=f"{self.tracker.total_yawns:,}")

        now = time.time()
        is_holding = (self.tracker.yawn_enabled and self.tracker.yawn_start_time is not None and
                      not self.tracker.already_counted_yawn)
        if is_holding:
            hold_sec = min(self.tracker.min_yawn_hold_sec, now - self.tracker.yawn_start_time)
            self.lbl_yawn_status.configure(
                text=f"😮 YAWNING... ({hold_sec:.1f}s / {self.tracker.min_yawn_hold_sec:.1f}s)",
                text_color=COLOR_PURPLE
            )
        elif self.tracker.is_yawning:
            self.lbl_yawn_status.configure(text="🟣 YAWN CONFIRMED!", text_color=COLOR_PURPLE)
        else:
            th_pct = int(self.tracker.yawn_threshold * 100)
            self.lbl_yawn_status.configure(text=f"Mouth: Normal (Threshold: {th_pct}%)", text_color=COLOR_GREEN)

        # Health & BPM
        bpm = self.tracker.get_bpm()
        self.lbl_bpm.configure(text=f"{bpm:.1f}")

        drowsy_text, drowsy_color_bgr = self.tracker.get_drowsiness_status()
        drowsy_hex = COLOR_GREEN if "ALERT" in drowsy_text else (COLOR_YELLOW if "TIRED" in drowsy_text else COLOR_RED)
        self.lbl_alertness.configure(text=drowsy_text, text_color=drowsy_hex)

        # Session & Blink timers
        elapsed_sec = int(time.time() - self.session_start_time)
        last_blink_sec = self.tracker.get_time_since_last_blink()
        self.lbl_session_time.configure(
            text=f"Session: {elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}   •   Last Blink: {last_blink_sec:.1f}s ago"
        )

        # FPS badge
        self.fps_badge.configure(text=f"{int(self.fps)} FPS")

        # Re-schedule every 500ms
        self.after(500, self._update_metrics_cards)

    def save_current_session_to_db(self):
        """Saves session duration and counts to SQLite."""
        elapsed_sec = int(time.time() - self.session_start_time)
        bpm = self.tracker.get_bpm()
        drowsy_text, _ = self.tracker.get_drowsiness_status()

        # Only save if there was actual tracking activity to avoid saving empty junk sessions
        if self.tracker.total_blinks > 0 or self.tracker.total_yawns > 0 or elapsed_sec >= 30:
            sid = self.db.save_session(
                start_time=self.session_start_time,
                end_time=time.time(),
                duration_seconds=elapsed_sec,
                total_blinks=self.tracker.total_blinks,
                total_yawns=self.tracker.total_yawns,
                left_winks=self.tracker.left_winks,
                right_winks=self.tracker.right_winks,
                avg_bpm=bpm,
                fatigue_status=drowsy_text
            )
            print(f"[Storage] Session #{sid} saved to SQLite database.")

    def on_close(self):
        """Handles application shutdown cleanly."""
        print("[Shutdown] Closing application...")
        if self.auto_save:
            self.save_current_session_to_db()

        self.stop_camera()
        self.destroy()
        sys.exit(0)


def main():
    app = OcularDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
