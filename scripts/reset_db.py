#!/usr/bin/env python3
"""Reset the Solidcue SQLite database.

Deletes all rows from user tables in the checkpoint DB while preserving the
table schema. This keeps LangGraph checkpoint tables available for reuse.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def resolve_db_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser()
    configured_path = os.getenv("SOLIDCUE_CHECKPOINT_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".solidcue" / "checkpoints.sqlite"


def list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    return [row[0] for row in cur.fetchall() if row and isinstance(row[0], str)]


def ensure_chat_history_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            agent_key TEXT,
            last_thread_id TEXT,
            last_run_id TEXT,
            last_run_status TEXT,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        )
        """
    )
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(conversations)")
    columns = {row[1] for row in cur.fetchall() if len(row) > 1 and isinstance(row[1], str)}
    if "worked_seconds" in columns:
        conn.execute("ALTER TABLE conversations RENAME TO conversations_legacy")
        conn.execute(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                agent_key TEXT,
                last_thread_id TEXT,
                last_run_id TEXT,
                last_run_status TEXT,
                created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
                updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO conversations (
                conversation_id,
                agent_key,
                last_thread_id,
                last_run_id,
                last_run_status,
                created_at,
                updated_at
            )
            SELECT
                conversation_id,
                agent_key,
                last_thread_id,
                last_run_id,
                last_run_status,
                created_at,
                updated_at
            FROM conversations_legacy
            """
        )
        conn.execute("DROP TABLE conversations_legacy")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            agent_key TEXT,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        )
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset the Solidcue SQLite database")
    parser.add_argument("--db-path", help="Path to the SQLite database")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that all rows should be deleted from user tables",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    print(f"DB path: {db_path}")

    if not db_path.exists():
        print("Database file does not exist")
        return 1

    with sqlite3.connect(str(db_path)) as conn:
        tables = list_tables(conn)
        ensure_chat_history_schema(conn)

        if not tables:
            print("No user tables found")
            conn.commit()
            return 0

        print("User tables to clear:")
        for table in tables:
            print(f"  - {table}")

        if not args.yes:
            print("Refusing to delete rows without --yes")
            return 1

        cur = conn.cursor()
        for table in tables:
            cur.execute(f'DELETE FROM "{table}"')

        try:
            cur.execute(
                "DELETE FROM sqlite_sequence WHERE name IN (%s)" % ", ".join("?" for _ in tables),
                tables,
            )
        except sqlite3.OperationalError:
            pass

        conn.commit()
        print("Cleared all rows from user tables")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
