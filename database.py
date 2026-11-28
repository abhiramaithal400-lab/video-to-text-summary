"""
Database management module for Sports Match Video to Text Summarization.
Uses SQLite (Relational SQL Database) for tracking match videos, ground-truth annotations,
detected highlight events, and generated summaries.
"""

import os
import sqlite3
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "sports_summary.db"
LABELS_CSV = BASE_DIR / "data" / "labels.csv"


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Establish a connection to the SQLite database with foreign keys enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Initialize database tables for videos, annotations, detected events, and summaries."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Video files table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        duration_seconds REAL DEFAULT 0.0,
        file_size_mb REAL DEFAULT 0.0,
        status TEXT DEFAULT 'PROCESSED',
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Ground-truth annotations table (replaces / mirrors labels.csv)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clip_name TEXT NOT NULL,
        time_seconds REAL NOT NULL,
        event TEXT NOT NULL
    );
    """)

    # 3. Detected events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detected_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        timestamp_seconds REAL NOT NULL,
        event_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)

    # 4. Summaries table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        summary_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
    );
    """)

    conn.commit()

    # Seed annotations if table is currently empty
    cursor.execute("SELECT COUNT(*) FROM annotations;")
    count = cursor.fetchone()[0]
    if count == 0 and LABELS_CSV.exists():
        try:
            df = pd.read_csv(LABELS_CSV)
            for _, row in df.iterrows():
                cursor.execute(
                    "INSERT INTO annotations (clip_name, time_seconds, event) VALUES (?, ?, ?);",
                    (str(row["clip_name"]).strip(), float(row["time_seconds"]), str(row["event"]).strip().lower())
                )
            conn.commit()
            print(f"[Database] Seeded {len(df)} annotations into SQLite from {LABELS_CSV.name}")
        except Exception as e:
            print(f"[Database] Warning: Could not seed annotations: {e}")

    # Seed demo match analyses if videos table is currently empty
    cursor.execute("SELECT COUNT(*) FROM videos;")
    vid_count = cursor.fetchone()[0]
    if vid_count == 0:
        demo_fixtures = [
            ("100.mp4", 74.0, 18.5, [(18.0, "goal", 0.95)], "18' - Goal"),
            ("101.mp4", 124.0, 31.0, [(35.0, "goal", 0.91)], "35' - Goal"),
            ("107.mp4", 126.0, 31.5, [(34.0, "corner", 0.88)], "34' - Corner"),
            ("10_demo.mp4", 90.0, 22.5, [(44.0, "goal", 0.94), (84.0, "foul", 0.82)], "44' - Goal\n84' - Foul"),
            ("1_demo.mp4", 120.0, 30.0, [(88.0, "goal", 0.93), (108.0, "card", 0.87)], "88' - Goal\n108' - Yellow card"),
        ]
        for fname, dur, size_mb, evs, summary in demo_fixtures:
            cursor.execute(
                "INSERT INTO videos (file_name, duration_seconds, file_size_mb, status) VALUES (?, ?, ?, 'PROCESSED');",
                (fname, dur, size_mb)
            )
            v_id = cursor.lastrowid
            for ts, ev_name, conf in evs:
                cursor.execute(
                    "INSERT INTO detected_events (video_id, timestamp_seconds, event_type, confidence) VALUES (?, ?, ?, ?);",
                    (v_id, ts, ev_name, conf)
                )
            cursor.execute(
                "INSERT INTO summaries (video_id, summary_text) VALUES (?, ?);",
                (v_id, summary)
            )
        conn.commit()
        print(f"[Database] Seeded {len(demo_fixtures)} match records into SQLite videos table.")

    conn.close()


def get_annotations_dataframe(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Query ground-truth annotations from SQLite and return as a Pandas DataFrame."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT clip_name, time_seconds, event FROM annotations ORDER BY id ASC;", conn)
        return df
    finally:
        conn.close()


def record_video_analysis(
    file_path: str,
    duration_seconds: float,
    events: list[tuple[float, str, float]],
    summary_text: str,
    db_path: Path | str = DEFAULT_DB_PATH
) -> int:
    """
    Record an analyzed video, its detected highlight events, and the generated summary.
    Returns the auto-generated video_id.
    """
    file_name = os.path.basename(file_path)
    file_size_mb = 0.0
    if os.path.exists(file_path):
        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)

    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO videos (file_name, duration_seconds, file_size_mb, status) VALUES (?, ?, ?, 'PROCESSED');",
            (file_name, round(duration_seconds, 2), file_size_mb)
        )
        video_id = cursor.lastrowid

        for ev in events:
            # ev format: (timestamp, label, confidence)
            ts = float(ev[0])
            lbl = str(ev[1])
            conf = float(ev[2]) if len(ev) > 2 else 1.0
            cursor.execute(
                "INSERT INTO detected_events (video_id, timestamp_seconds, event_type, confidence) VALUES (?, ?, ?, ?);",
                (video_id, ts, lbl, conf)
            )

        cursor.execute(
            "INSERT INTO summaries (video_id, summary_text) VALUES (?, ?);",
            (video_id, summary_text)
        )

        conn.commit()
        return video_id
    finally:
        conn.close()


def get_all_records(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    """Retrieve all processed videos along with their detected events and summaries."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT v.id, v.file_name, v.duration_seconds, v.file_size_mb, v.processed_at, s.summary_text
        FROM videos v
        LEFT JOIN summaries s ON v.id = s.video_id
        ORDER BY v.id DESC;
        """)
        rows = cursor.fetchall()
        results = []
        for r in rows:
            vid_id = r["id"]
            cursor.execute("SELECT timestamp_seconds, event_type, confidence FROM detected_events WHERE video_id = ? ORDER BY timestamp_seconds ASC;", (vid_id,))
            evs = [dict(ev) for ev in cursor.fetchall()]
            results.append({
                "id": vid_id,
                "file_name": r["file_name"],
                "duration_seconds": r["duration_seconds"],
                "file_size_mb": r["file_size_mb"],
                "processed_at": r["processed_at"],
                "summary": r["summary_text"] or "No summary",
                "events": evs
            })
        return results
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("[Database] Schema verified and ready.")
