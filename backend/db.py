import contextlib
import logging
import sqlite3
from pathlib import Path
from typing import Iterable

from ml.config import DB_PATH, ensure_directories

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  model_name TEXT NOT NULL,
  input_mode TEXT NOT NULL,
  input_path TEXT NOT NULL,
  pre_path TEXT NOT NULL,
  reference_post_path TEXT,
  generated_post_path TEXT NOT NULL,
  status TEXT NOT NULL,
  notes TEXT DEFAULT ''
);
"""

_JOURNAL_MODE_APPLIED = False


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply PRAGMAs on a new connection.

    `journal_mode=WAL` is a persistent database-file property and only needs
    to be set once per process lifetime, but `busy_timeout` and `synchronous`
    are per-connection attributes that default back to 0 / FULL on every new
    connection, so they MUST be re-applied each time.
    """
    global _JOURNAL_MODE_APPLIED
    try:
        # Per-connection settings (must run on every new connection).
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        # WAL mode: persistent file-level property. Apply once. sqlite3's
        # Python driver may start an implicit transaction; roll back first so
        # journal_mode=WAL is accepted.
        if not _JOURNAL_MODE_APPLIED:
            conn.rollback()
            conn.execute("PRAGMA journal_mode=WAL")
            _JOURNAL_MODE_APPLIED = True
    except sqlite3.DatabaseError as exc:
        logger.warning("Failed to apply SQLite pragmas: %s", exc)


@contextlib.contextmanager
def _connection():
    """Context-managed SQLite connection. Commits on success, rolls back on error."""
    ensure_directories()
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        _apply_pragmas(conn)
        conn.execute(SCHEMA)
        conn.commit()
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def connect() -> sqlite3.Connection:
    """Deprecated: legacy direct connection. Prefer _connection() context manager."""
    ensure_directories()
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_history(record: dict) -> int:
    with _connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_history (
              created_at, model_name, input_mode, input_path, pre_path,
              reference_post_path, generated_post_path, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["created_at"],
                record["model_name"],
                record["input_mode"],
                record["input_path"],
                record["pre_path"],
                record.get("reference_post_path"),
                record["generated_post_path"],
                record["status"],
                record.get("notes", ""),
            ),
        )
        return int(cursor.lastrowid)


def fetch_history() -> Iterable[dict]:
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, model_name, input_mode, input_path, pre_path,
                   reference_post_path, generated_post_path, status, notes
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]
