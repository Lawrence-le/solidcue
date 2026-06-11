from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
import os


def resolve_checkpoint_db_path() -> Path:
    configured_path = os.getenv("SOLIDCUE_CHECKPOINT_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".solidcue" / "checkpoints.sqlite"


def _connect() -> sqlite3.Connection:
    db_path = resolve_checkpoint_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def _ensure_schema(conn: sqlite3.Connection) -> None:
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
        _migrate_conversations_schema(conn, columns)
        cur.execute("PRAGMA table_info(conversations)")
        columns = {row[1] for row in cur.fetchall() if len(row) > 1 and isinstance(row[1], str)}
    if "last_thread_id" not in columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN last_thread_id TEXT")
    if "last_run_id" not in columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN last_run_id TEXT")
    if "last_run_status" not in columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN last_run_status TEXT")
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
    column = _chat_history_id_column(conn)
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_chat_history_{column}_id ON chat_history({column}, id)"
    )


def _migrate_conversations_schema(conn: sqlite3.Connection, columns: set[str]) -> None:
    preserved_columns = [
        column
        for column in (
            "conversation_id",
            "agent_key",
            "last_thread_id",
            "last_run_id",
            "last_run_status",
            "created_at",
            "updated_at",
        )
        if column in columns
    ]
    if not preserved_columns:
        return

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
    select_list = ", ".join(preserved_columns)
    insert_list = ", ".join(preserved_columns)
    conn.execute(
        f"""
        INSERT INTO conversations ({insert_list})
        SELECT {select_list}
        FROM conversations_legacy
        """
    )
    conn.execute("DROP TABLE conversations_legacy")


def _upsert_conversation(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    agent_key: str | None = None,
    last_thread_id: str | None = None,
    last_run_id: str | None = None,
    last_run_status: str | None = None,
) -> None:
    normalized_agent_key = (
        agent_key.strip() if isinstance(agent_key, str) and agent_key.strip() else None
    )
    normalized_thread_id = (
        last_thread_id.strip()
        if isinstance(last_thread_id, str) and last_thread_id.strip()
        else None
    )
    normalized_run_id = (
        last_run_id.strip()
        if isinstance(last_run_id, str) and last_run_id.strip()
        else None
    )
    normalized_run_status = (
        last_run_status.strip()
        if isinstance(last_run_status, str) and last_run_status.strip()
        else None
    )
    conn.execute(
        """
        INSERT INTO conversations (
            conversation_id,
            agent_key,
            last_thread_id,
            last_run_id,
            last_run_status
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
            agent_key = COALESCE(excluded.agent_key, conversations.agent_key),
            last_thread_id = COALESCE(excluded.last_thread_id, conversations.last_thread_id),
            last_run_id = COALESCE(excluded.last_run_id, conversations.last_run_id),
            last_run_status = COALESCE(excluded.last_run_status, conversations.last_run_status),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            conversation_id,
            normalized_agent_key,
            normalized_thread_id,
            normalized_run_id,
            normalized_run_status,
        ),
    )


def upsert_conversation(
    *,
    conversation_id: str,
    agent_key: str | None = None,
    last_thread_id: str | None = None,
    last_run_id: str | None = None,
    last_run_status: str | None = None,
) -> None:
    normalized_conversation_id = (
        conversation_id.strip() if isinstance(conversation_id, str) else ""
    )
    if not normalized_conversation_id:
        return

    with _connect() as conn:
        _ensure_schema(conn)
        _upsert_conversation(
            conn,
            conversation_id=normalized_conversation_id,
            agent_key=agent_key,
            last_thread_id=last_thread_id,
            last_run_id=last_run_id,
            last_run_status=last_run_status,
        )
        conn.commit()


def update_conversation_run_state(
    *,
    conversation_id: str,
    agent_key: str | None = None,
    last_thread_id: str | None = None,
    last_run_id: str | None = None,
    last_run_status: str | None = None,
) -> None:
    normalized_conversation_id = (
        conversation_id.strip() if isinstance(conversation_id, str) else ""
    )
    if not normalized_conversation_id:
        return

    with _connect() as conn:
        _ensure_schema(conn)
        _upsert_conversation(
            conn,
            conversation_id=normalized_conversation_id,
            agent_key=agent_key,
            last_thread_id=last_thread_id,
            last_run_id=last_run_id,
            last_run_status=last_run_status,
        )
        conn.commit()


def get_conversation_metadata(conversation_id: str) -> dict[str, Any] | None:
    normalized_conversation_id = (
        conversation_id.strip() if isinstance(conversation_id, str) else ""
    )
    if not normalized_conversation_id:
        return None

    with _connect() as conn:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT conversation_id, agent_key, created_at, updated_at
                 , last_thread_id, last_run_id, last_run_status
            FROM conversations
            WHERE conversation_id = ?
            LIMIT 1
            """,
            (normalized_conversation_id,),
        )
        row = cur.fetchone()

    if not row:
        return None
    return {
        "conversation_id": row[0],
        "agent_key": row[1],
        "worked_seconds": 0,
        "created_at": row[2],
        "updated_at": row[3],
        "last_thread_id": row[4],
        "last_run_id": row[5],
        "last_run_status": row[6],
    }


def _chat_history_id_column(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(chat_history)")
        columns = [row[1] for row in cur.fetchall() if len(row) > 1 and isinstance(row[1], str)]
        if "conversation_id" in columns:
            return "conversation_id"
        if "thread_id" in columns:
            return "thread_id"
    except Exception:
        pass
    return "conversation_id"


def append_chat_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    agent_key: str | None = None,
) -> None:
    try:
        normalized_conversation_id = conversation_id.strip()
        normalized_role = role.strip()
        normalized_content = content.strip()
        if not normalized_conversation_id or not normalized_role or not normalized_content:
            return

        with _connect() as conn:
            _ensure_schema(conn)
            _upsert_conversation(
                conn,
                conversation_id=normalized_conversation_id,
                agent_key=agent_key,
            )
            column = _chat_history_id_column(conn)
            conn.execute(
                """
                INSERT INTO chat_history ({column}, role, content, agent_key)
                VALUES (?, ?, ?, ?)
                """.format(column=column),
                (normalized_conversation_id, normalized_role, normalized_content, agent_key),
            )
            conn.commit()
    except Exception:
        return


def load_chat_history(conversation_id: str | None, *, limit: int | None = None) -> list[dict[str, Any]]:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    if not normalized_conversation_id:
        return []

    with _connect() as conn:
        _ensure_schema(conn)
        column = _chat_history_id_column(conn)
        cur = conn.cursor()
        if isinstance(limit, int) and limit > 0:
            cur.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE {column} = ?
                ORDER BY id DESC
                LIMIT ?
                """.format(column=column),
                (normalized_conversation_id, limit),
            )
            rows = list(reversed(cur.fetchall()))
        else:
            cur.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE {column} = ?
                ORDER BY id ASC
                """.format(column=column),
                (normalized_conversation_id,),
            )
            rows = cur.fetchall()

    history: list[dict[str, Any]] = []
    for role, content in rows:
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        normalized_role = role.strip()
        normalized_content = content.strip()
        if not normalized_role or not normalized_content:
            continue
        history.append({"role": normalized_role, "content": normalized_content})
    return history


def delete_chat_history(conversation_id: str) -> bool:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    if not normalized_conversation_id:
        return False

    with _connect() as conn:
        _ensure_schema(conn)
        column = _chat_history_id_column(conn)
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM chat_history WHERE {column} = ?",
            (normalized_conversation_id,),
        )
        deleted = cur.rowcount > 0
        if deleted:
            conn.commit()
        else:
            conn.rollback()
        return deleted


def delete_conversation_metadata(conversation_id: str) -> bool:
    normalized_conversation_id = (
        conversation_id.strip() if isinstance(conversation_id, str) else ""
    )
    if not normalized_conversation_id:
        return False

    with _connect() as conn:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (normalized_conversation_id,),
        )
        deleted = cur.rowcount > 0
        if deleted:
            conn.commit()
        else:
            conn.rollback()
        return deleted
