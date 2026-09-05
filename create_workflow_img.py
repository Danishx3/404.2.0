#!/usr/bin/env python3
"""
Script to generate the high-resolution architecture and pipeline workflow diagram
for Ocular AI (saved as workflow.png).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def generate_diagram():
    # Dimensions & DPI
    fig_width = 20
    fig_height = 10.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
    
    # Background color
    bg_color = "#0B0E17"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Coordinate system 0 to 100 in X and Y
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    
    # --------------------------------------------------------------------------
    # Header Section
    # --------------------------------------------------------------------------
    # Subtle badge
    badge_box = FancyBboxPatch(
        (3.5, 93.5), 23, 3.2,
        boxstyle="round,pad=0.5,rounding_size=1.2",
        facecolor="#161F33", edgecolor="#00E5FF", linewidth=1.2
    )
    ax.add_patch(badge_box)
    ax.text(15, 94.8, "OCULAR AI  •  SYSTEM PIPELINE", color="#00E5FF",
            fontsize=10.5, fontweight="bold", ha="center", va="center", family="sans-serif")
    
    # Main Title & Subtitle
    ax.text(3.5, 89.2, "End-to-End Neural Detection & Storage Architecture",
            color="#F8FAFC", fontsize=22, fontweight="bold", ha="left", va="center", family="sans-serif")
    ax.text(3.5, 85.5, "Real-time 30 FPS webcam processing, MediaPipe FaceLandmarker, speech-immune ensemble filter, and SQLite analytics",
            color="#94A3B8", fontsize=11.5, ha="left", va="center", family="sans-serif")

    # --------------------------------------------------------------------------
    # Pipeline Stages (5 Horizontal Cards)
    # --------------------------------------------------------------------------
    stages = [
        {
            "num": "01",
            "title": "VIDEO CAPTURE",
            "tech": "OpenCV Engine",
            "color": "#00E5FF",
            "bg": "#121826",
            "border": "#1E293B",
            "items": [
                ("Input Source", "HD Webcam (Device 0/1)"),
                ("Capture Rate", "30 FPS @ 1280x720"),
                ("Color Space", "BGR to RGB Matrix"),
                ("Frame Mirror", "Horizontal cv2.flip(1)"),
                ("Worker Thread", "Non-blocking ThreadQueue"),
                ("Latency Cap", "Zero frame accumulation")
            ]
        },
        {
            "num": "02",
            "title": "NEURAL INFERENCE",
            "tech": "MediaPipe Tasks",
            "color": "#38BDF8",
            "bg": "#121826",
            "border": "#1E293B",
            "items": [
                ("Core Model", "face_landmarker.task"),
                ("Delegate", "XNNPACK CPU Accel"),
                ("Dense Mesh", "478 3D Surface Points"),
                ("Blendshapes", "52 Facial Muscle Weights"),
                ("Eyelid Blend", "eyeBlinkLeft & Right"),
                ("Jaw Opening", "jawOpen Blendshape")
            ]
        },
        {
            "num": "03",
            "title": "FEATURE EXTRACTION",
            "tech": "Hybrid MAR + EAR",
            "color": "#C084FC",
            "bg": "#121826",
            "border": "#1E293B",
            "items": [
                ("Inner Lip MAR", "||p13 - p14|| / ||p61 - p291||"),
                ("Lip Contours", "Outer (20pts) Inner (20pts)"),
                ("Mouth Score", "max(jawOpen, MAR * 1.10)"),
                ("Debounce EAR", "Continuous 60s Window"),
                ("Iris Tracking", "Pupil Centers 468 & 473"),
                ("Noise Filter", "Micro-jitter rejection")
            ]
        },
        {
            "num": "04",
            "title": "STATE MACHINE",
            "tech": "Speech-Immune Filter",
            "color": "#FBBF24",
            "bg": "#121826",
            "border": "#1E293B",
            "items": [
                ("Blink Debounce", "40ms - 600ms Hysteresis"),
                ("Wink Detection", "Independent Left / Right"),
                ("Speech Filter", "15% - 35% mouth ignored"),
                ("Yawn Trigger", "Wide mouth >= 58%"),
                ("Continuous Hold", ">= 1.30s uninterrupted"),
                ("Flutter Reset", "Instant reset on talking")
            ]
        },
        {
            "num": "05",
            "title": "OUTPUT & STORAGE",
            "tech": "CustomTkinter + SQLite",
            "color": "#10B981",
            "bg": "#121826",
            "border": "#1E293B",
            "items": [
                ("GUI Viewport", "Centered letterbox label"),
                ("Sidebar (400px)", "Blink, Yawn & BPM cards"),
                ("Audio Daemon", "winsound 1400Hz/680Hz"),
                ("SQLite Database", "data/blink_history.db"),
                ("Analytics View", "Lifetime stats & CSV export"),
                ("Safe Clear", "Atomic confirmation wipe")
            ]
        }
    ]

    card_width = 16.8
    card_height = 64
    card_y = 17
    card_spacing = 2.4
    start_x = 3.5

    for i, s in enumerate(stages):
        x = start_x + i * (card_width + card_spacing)
        
        # Outer Card Box
        card = FancyBboxPatch(
            (x, card_y), card_width, card_height,
            boxstyle="round,pad=0.6,rounding_size=1.5",
            facecolor=s["bg"], edgecolor=s["color"], linewidth=1.5
        )
        ax.add_patch(card)
        
        # Inner Card Header Background
        header_patch = FancyBboxPatch(
            (x + 0.4, card_y + card_height - 10.5), card_width - 0.8, 9.5,
            boxstyle="round,pad=0.4,rounding_size=1.0",
            facecolor="#1A2234", edgecolor="none"
        )
        ax.add_patch(header_patch)

        # Stage Number Badge
        num_badge = FancyBboxPatch(
            (x + 1.2, card_y + card_height - 7.5), 3.2, 4.2,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            facecolor=s["color"], edgecolor="none"
        )
        ax.add_patch(num_badge)
        ax.text(x + 2.8, card_y + card_height - 5.4, s["num"],
                color="#0B0E17", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif")
        
        # Stage Titles
        ax.text(x + 5.2, card_y + card_height - 4.2, s["title"],
                color="#F8FAFC", fontsize=10.5, fontweight="bold", ha="left", va="center", family="sans-serif")
        ax.text(x + 5.2, card_y + card_height - 7.5, s["tech"],
                color=s["color"], fontsize=9, fontweight="bold", ha="left", va="center", family="sans-serif")

        # Feature List
        cur_y = card_y + card_height - 14.5
        for label, val in s["items"]:
            # Bullet Dot
            ax.plot(x + 1.6, cur_y + 0.5, marker="o", markersize=4, color=s["color"])
            # Label
            ax.text(x + 2.8, cur_y + 1.2, label, color="#F1F5F9", fontsize=9.2, fontweight="bold", family="sans-serif")
            # Description
            ax.text(x + 2.8, cur_y - 1.8, val, color="#94A3B8", fontsize=8.2, family="sans-serif")
            cur_y -= 8.2

        # Draw connecting flow arrows between cards
        if i < len(stages) - 1:
            arrow_x_start = x + card_width + 0.5
            arrow_x_end = arrow_x_start + card_spacing - 1.0
            arrow_y = card_y + card_height / 2
            
            arrow = FancyArrowPatch(
                (arrow_x_start, arrow_y), (arrow_x_end, arrow_y),
                arrowstyle="-|>", mutation_scale=14,
                color="#00E5FF", linewidth=2.0
            )
            ax.add_patch(arrow)

    # --------------------------------------------------------------------------
    # Bottom Technical Spec Ribbon
    # --------------------------------------------------------------------------
    ribbon_box = FancyBboxPatch(
        (3.5, 3.8), 93, 7.5,
        boxstyle="round,pad=0.5,rounding_size=1.2",
        facecolor="#121826", edgecolor="#232A3B", linewidth=1.2
    )
    ax.add_patch(ribbon_box)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

    specs = [
        ("• Real-Time Ingestion", "30 FPS @ 720p with threading"),
        ("• Neural Landmarks", "478 3D points + 52 blendshapes"),
        ("• Speech-Immunity", "58% threshold + 1.3s hold"),
        ("• Fatigue Analytics", "Rolling BPM & Drowsiness status"),
        ("• SQLite Storage", "Auto-save & CSV export")
    ]

    rx = 5.5
    for title, desc in specs:
        ax.text(rx, 8.4, title, color="#00E5FF", fontsize=9.8, fontweight="bold", family="sans-serif")
        ax.text(rx, 5.6, desc, color="#94A3B8", fontsize=8.5, family="sans-serif")
        rx += 18.6
        if rx < 90:
            ax.plot([rx - 2.5, rx - 2.5], [4.8, 10.2], color="#232A3B", linewidth=1.5)

    plt.tight_layout()
    output_path = "workflow.png"
    plt.savefig(output_path, facecolor=bg_color, edgecolor="none", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Success] Workflow diagram generated: {output_path}")

if __name__ == "__main__":
    generate_diagram()
