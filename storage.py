"""
Database and Local Storage Engine for Eye & Yawn Tracker AI
------------------------------------------------------------
Uses SQLite to persist tracking sessions, lifetime totals, and session history locally.
"""

import os
import csv
import sqlite3
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "blink_history.db")


class StorageManager:
    """Manages SQLite database operations for ocular tracking sessions."""
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    total_blinks INTEGER NOT NULL,
                    total_yawns INTEGER NOT NULL,
                    left_winks INTEGER NOT NULL,
                    right_winks INTEGER NOT NULL,
                    avg_bpm REAL NOT NULL,
                    fatigue_status TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_session(self, start_time, end_time, duration_seconds, total_blinks,
                     total_yawns, left_winks, right_winks, avg_bpm, fatigue_status):
        """Saves a finished session into the SQLite database."""
        # Only save meaningful sessions (> 3 seconds or with at least 1 event)
        if duration_seconds < 3 and total_blinks == 0 and total_yawns == 0:
            return None

        # Format timestamps nicely
        if isinstance(start_time, (int, float)):
            start_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_str = str(start_time)

        if isinstance(end_time, (int, float)):
            end_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
        else:
            end_str = str(end_time)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (
                    start_time, end_time, duration_seconds, total_blinks,
                    total_yawns, left_winks, right_winks, avg_bpm, fatigue_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (start_str, end_str, int(duration_seconds), int(total_blinks),
                  int(total_yawns), int(left_winks), int(right_winks),
                  float(avg_bpm), str(fatigue_status)))
            conn.commit()
            return cursor.lastrowid

    def get_lifetime_stats(self):
        """Returns aggregated lifetime statistics across all recorded sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(id) as total_sessions,
                    COALESCE(SUM(total_blinks), 0) as lifetime_blinks,
                    COALESCE(SUM(total_yawns), 0) as lifetime_yawns,
                    COALESCE(SUM(left_winks + right_winks), 0) as lifetime_winks,
                    COALESCE(SUM(duration_seconds), 0) as lifetime_seconds,
                    COALESCE(AVG(avg_bpm), 0) as avg_bpm
                FROM sessions
            """)
            row = cursor.fetchone()
            if row:
                total_sec = row[4]
                hours = total_sec / 3600.0
                return {
                    "total_sessions": row[0],
                    "lifetime_blinks": row[1],
                    "lifetime_yawns": row[2],
                    "lifetime_winks": row[3],
                    "lifetime_hours": round(hours, 2),
                    "lifetime_minutes": round(total_sec / 60.0, 1),
                    "avg_bpm": round(row[5], 1)
                }
            return {
                "total_sessions": 0,
                "lifetime_blinks": 0,
                "lifetime_yawns": 0,
                "lifetime_winks": 0,
                "lifetime_hours": 0.0,
                "lifetime_minutes": 0.0,
                "avg_bpm": 0.0
            }

    def get_recent_sessions(self, limit=50):
        """Retrieves recent sessions ordered by start_time descending."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, start_time, duration_seconds, total_blinks, total_yawns,
                       (left_winks + right_winks) as total_winks, avg_bpm, fatigue_status
                FROM sessions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def export_to_csv(self, output_path=None):
        """Exports all sessions into a CSV file."""
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(DB_DIR, f"blink_tracker_export_{ts}.csv")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, start_time, end_time, duration_seconds, total_blinks,
                       total_yawns, left_winks, right_winks, avg_bpm, fatigue_status
                FROM sessions ORDER BY id ASC
            """)
            rows = cursor.fetchall()

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Session ID", "Start Time", "End Time", "Duration (Seconds)",
                "Total Blinks", "Total Yawns", "Left Winks", "Right Winks",
                "Average BPM", "Fatigue Assessment"
            ])
            writer.writerows(rows)

        return output_path

    def clear_history(self):
        """Clears all session logs from the database safely."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions")
            conn.commit()
            conn.isolation_level = None
            cursor.execute("VACUUM")
        finally:
            conn.close()
