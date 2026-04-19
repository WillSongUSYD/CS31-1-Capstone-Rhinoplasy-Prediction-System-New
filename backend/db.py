import contextlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from ml.config import DB_PATH, PREDICTIONS_DIR, ensure_directories

logger = logging.getLogger(__name__)


def _to_relative(path: Optional[str]) -> Optional[str]:
    """Convert an absolute path under PREDICTIONS_DIR to a relative one.

    Returns the input unchanged if it's already relative, None, or points
    outside PREDICTIONS_DIR (we can't safely shorten those).
    """
    if not path:
        return path
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.resolve().relative_to(PREDICTIONS_DIR.resolve()))
    except (ValueError, OSError):
        return path


def _absolutize(path: Optional[str]) -> Optional[str]:
    """Re-attach PREDICTIONS_DIR to a stored relative path.

    Absolute paths (legacy records) are returned unchanged so both formats
    coexist during/after migration.
    """
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(PREDICTIONS_DIR / p)


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
# Declared alongside _JOURNAL_MODE_APPLIED so both exist before the
# at-fork handler is registered below (the handler mutates both).
_PATHS_MIGRATED = False


def _reset_journal_flag_after_fork() -> None:
    """Reset module-level one-shot flags in forked workers so they reapply
    the PRAGMAs and path migration on their first connection. Both operations
    are idempotent, so re-running them in a child is harmless.
    """
    global _JOURNAL_MODE_APPLIED, _PATHS_MIGRATED
    _JOURNAL_MODE_APPLIED = False
    _PATHS_MIGRATED = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_journal_flag_after_fork)


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


def _migrate_paths_to_relative(conn: sqlite3.Connection) -> None:
    """One-shot migration: rewrite absolute paths under PREDICTIONS_DIR to relative.

    Idempotent (it's a no-op on rows already stored as relative paths) and
    safe to call on every process start. Only paths demonstrably inside
    PREDICTIONS_DIR are rewritten; rows that escaped the predictions dir
    remain absolute so no data is lost.
    """
    global _PATHS_MIGRATED
    if _PATHS_MIGRATED:
        return
    try:
        pred_root = PREDICTIONS_DIR.resolve()
        rows = conn.execute(
            "SELECT id, input_path, pre_path, reference_post_path, generated_post_path "
            "FROM prediction_history"
        ).fetchall()
        for row in rows:
            updates: dict = {}
            for col in ("input_path", "pre_path", "reference_post_path", "generated_post_path"):
                original = row[col]
                if not original:
                    continue
                if not Path(original).is_absolute():
                    continue  # already relative
                # Only shorten paths under PREDICTIONS_DIR - don't mangle
                # anything that happens to live elsewhere on disk. Use
                # relative_to instead of a string-prefix check to avoid
                # "/foo/bar/predictions-backup" matching "/foo/bar/predictions".
                try:
                    resolved = Path(original).resolve()
                except OSError:
                    continue
                try:
                    resolved.relative_to(pred_root)
                except ValueError:
                    continue  # not under PREDICTIONS_DIR, keep absolute
                new = _to_relative(original)
                if new is not None and new != original:
                    updates[col] = new
            if updates:
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                conn.execute(
                    f"UPDATE prediction_history SET {set_clause} WHERE id = ?",
                    (*updates.values(), row["id"]),
                )
        conn.commit()
    except sqlite3.DatabaseError as exc:
        logger.warning("Path migration skipped: %s", exc)
    else:
        # Only flip the one-shot flag on success so a partial/failed migration
        # gets retried on the next connection instead of being marked done.
        _PATHS_MIGRATED = True


@contextlib.contextmanager
def _connection():
    """Context-managed SQLite connection. Commits on success, rolls back on error."""
    ensure_directories()
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        _apply_pragmas(conn)
        conn.execute(SCHEMA)
        _migrate_paths_to_relative(conn)
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
    # Store paths relative to PREDICTIONS_DIR so the record survives moving
    # the predictions directory (e.g. when deploying or backing up).
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
                _to_relative(record["input_path"]),
                _to_relative(record["pre_path"]),
                _to_relative(record.get("reference_post_path")),
                _to_relative(record["generated_post_path"]),
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
    # Rehydrate paths to absolute on the way out so callers (e.g. the backend
    # URL builder) don't need to know whether the row was stored relative
    # or absolute.
    result = []
    for row in rows:
        d = dict(row)
        for col in ("input_path", "pre_path", "reference_post_path", "generated_post_path"):
            d[col] = _absolutize(d.get(col))
        result.append(d)
    return result
