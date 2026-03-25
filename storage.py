import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'userdata.db')


def _conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS winners (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT NOT NULL,
            data      TEXT NOT NULL,
            saved_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def save_winner(user_id: str, winner: dict):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO winners (user_id, data) VALUES (?, ?)",
            (user_id, json.dumps(winner)),
        )


def load_winners(user_id: str) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT data FROM winners WHERE user_id = ? ORDER BY saved_at",
            (user_id,),
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def delete_winners(user_id: str):
    with _conn() as conn:
        conn.execute("DELETE FROM winners WHERE user_id = ?", (user_id,))
