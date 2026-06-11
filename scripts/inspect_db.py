#!/usr/bin/env python3
"""Inspect the Solidcue SQLite database.

Shows table names, row counts, and recent rows from the checkpoint and chat
history tables. This is intentionally read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
import struct
from typing import Any


def resolve_db_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser()
    configured_path = os.getenv("SOLIDCUE_CHECKPOINT_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".solidcue" / "checkpoints.sqlite"


def connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    return [row[0] for row in cur.fetchall() if row and isinstance(row[0], str)]


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table_name}")')
    return [row[1] for row in cur.fetchall() if len(row) > 1 and isinstance(row[1], str)]


def row_count(conn: sqlite3.Connection, table_name: str) -> int:
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.hex()
    return value


def parse_jsonish(value: Any) -> Any:
    value = decode_value(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def parse_checkpoint_payload(value: Any) -> Any:
    parsed = parse_jsonish(value)
    if isinstance(parsed, dict):
        return parsed

    raw = value if isinstance(value, bytes) else None
    if raw:
        try:
            import msgpack  # type: ignore

            unpacked = msgpack.unpackb(raw, raw=False)
            if isinstance(unpacked, dict):
                return unpacked
        except Exception:
            try:
                unpacked = decode_msgpack(raw)
                if isinstance(unpacked, dict):
                    return unpacked
            except Exception:
                pass

    return parsed


def decode_msgpack(data: bytes) -> Any:
    """Decode the msgpack subset used by LangGraph checkpoints.

    This intentionally supports the common scalar, array, and map types
    needed for checkpoint state inspection without requiring an extra
    dependency at runtime.
    """

    idx = 0

    def read(n: int) -> bytes:
        nonlocal idx
        chunk = data[idx : idx + n]
        if len(chunk) != n:
            raise ValueError("truncated msgpack payload")
        idx += n
        return chunk

    def decode() -> Any:
        nonlocal idx
        prefix = read(1)[0]

        if prefix <= 0x7F:
            return prefix
        if 0x80 <= prefix <= 0x8F:
            size = prefix & 0x0F
            return {decode(): decode() for _ in range(size)}
        if 0x90 <= prefix <= 0x9F:
            size = prefix & 0x0F
            return [decode() for _ in range(size)]
        if 0xA0 <= prefix <= 0xBF:
            size = prefix & 0x1F
            return read(size).decode("utf-8", errors="replace")
        if prefix == 0xC0:
            return None
        if prefix == 0xC2:
            return False
        if prefix == 0xC3:
            return True
        if prefix == 0xC4:
            size = read(1)[0]
            return read(size)
        if prefix == 0xC5:
            size = struct.unpack(">H", read(2))[0]
            return read(size)
        if prefix == 0xC6:
            size = struct.unpack(">I", read(4))[0]
            return read(size)
        if prefix == 0xCA:
            return struct.unpack(">f", read(4))[0]
        if prefix == 0xCB:
            return struct.unpack(">d", read(8))[0]
        if prefix == 0xCC:
            return read(1)[0]
        if prefix == 0xCD:
            return struct.unpack(">H", read(2))[0]
        if prefix == 0xCE:
            return struct.unpack(">I", read(4))[0]
        if prefix == 0xCF:
            return struct.unpack(">Q", read(8))[0]
        if prefix == 0xD0:
            return struct.unpack(">b", read(1))[0]
        if prefix == 0xD1:
            return struct.unpack(">h", read(2))[0]
        if prefix == 0xD2:
            return struct.unpack(">i", read(4))[0]
        if prefix == 0xD3:
            return struct.unpack(">q", read(8))[0]
        if prefix == 0xD9:
            size = read(1)[0]
            return read(size).decode("utf-8", errors="replace")
        if prefix == 0xDA:
            size = struct.unpack(">H", read(2))[0]
            return read(size).decode("utf-8", errors="replace")
        if prefix == 0xDB:
            size = struct.unpack(">I", read(4))[0]
            return read(size).decode("utf-8", errors="replace")
        if prefix == 0xDC:
            size = struct.unpack(">H", read(2))[0]
            return [decode() for _ in range(size)]
        if prefix == 0xDD:
            size = struct.unpack(">I", read(4))[0]
            return [decode() for _ in range(size)]
        if prefix == 0xDE:
            size = struct.unpack(">H", read(2))[0]
            return {decode(): decode() for _ in range(size)}
        if prefix == 0xDF:
            size = struct.unpack(">I", read(4))[0]
            return {decode(): decode() for _ in range(size)}
        if 0xE0 <= prefix <= 0xFF:
            return prefix - 0x100
        raise ValueError(f"unsupported msgpack prefix: 0x{prefix:02x}")

    result = decode()
    return result


def extract_state_fields(payload: Any, keys: list[str]) -> dict[str, Any]:
    def _find(value: Any, key: str) -> tuple[bool, Any]:
        if isinstance(value, dict):
            if key in value:
                return True, value.get(key)
            for child in value.values():
                found, child_value = _find(child, key)
                if found:
                    return True, child_value
        elif isinstance(value, list):
            for child in value:
                found, child_value = _find(child, key)
                if found:
                    return True, child_value
        return False, None

    result: dict[str, Any] = {}
    for key in keys:
        found, value = _find(payload, key)
        result[key] = value if found else None
    return result


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def truncate_text(value: Any, max_len: int = 80) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    string_rows = [[truncate_text(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def render_row(row: list[str]) -> str:
        cells = [f" {cell.ljust(widths[idx])} " for idx, cell in enumerate(row)]
        return "│" + "│".join(cells) + "│"

    print(border("┌", "┬", "┐"))
    print(render_row(headers))
    print(border("├", "┼", "┤"))
    for row in string_rows:
        print(render_row(row))
    print(border("└", "┴", "┘"))


def print_table_summary(conn: sqlite3.Connection, table_name: str) -> None:
    columns = table_columns(conn, table_name)
    count = row_count(conn, table_name)
    rows = [[table_name, count, ", ".join(columns) if columns else ""]]
    print_table(["table", "rows", "columns"], rows)


def print_checkpoint_rows(
    conn: sqlite3.Connection,
    *,
    limit: int,
    thread_id: str | None,
    conversation_id: str | None,
) -> None:
    if "checkpoints" not in list_tables(conn):
        print("checkpoints: <no rows>")
        return

    columns = set(table_columns(conn, "checkpoints"))
    if "thread_id" not in columns:
        print("checkpoints: <no rows>")
        return

    filters: list[str] = []
    params: list[Any] = []
    if thread_id:
        filters.append("thread_id = ?")
        params.append(thread_id)
    if conversation_id and "metadata" in columns:
        filters.append("json_extract(metadata, '$.conversation_id') = ?")
        params.append(conversation_id)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
        SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        FROM checkpoints
        {where_clause}
        ORDER BY rowid DESC
        LIMIT ?
    """
    params.append(limit)

    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()

    if not rows:
        print("checkpoints: <no rows>")
        return

    table_rows: list[list[Any]] = []
    for idx, row in enumerate(rows, start=1):
        thread, namespace, checkpoint_id, parent_id, row_type, checkpoint, metadata = row
        metadata_obj = parse_jsonish(metadata)
        checkpoint_obj = parse_checkpoint_payload(checkpoint)
        conversation = None
        agent_key = None
        step = None
        if isinstance(metadata_obj, dict):
            conversation = metadata_obj.get("conversation_id")
            agent_key = metadata_obj.get("agent_key")
            step = metadata_obj.get("step")
        checkpoint_fields = extract_state_fields(
            checkpoint_obj,
            ["worked_seconds", "timer_started_at", "phase", "current_task"],
        )
        table_rows.append(
            [
                idx,
                decode_value(thread),
                decode_value(namespace),
                decode_value(checkpoint_id),
                decode_value(parent_id),
                decode_value(row_type),
                conversation,
                agent_key,
                step,
                checkpoint_fields["worked_seconds"],
                checkpoint_fields["timer_started_at"],
                checkpoint_fields["phase"],
                checkpoint_fields["current_task"],
            ]
        )

    print_table(
        [
            "#",
            "thread_id",
            "ns",
            "checkpoint_id",
            "parent_id",
            "type",
            "conversation_id",
            "agent_key",
            "step",
            "worked_seconds",
            "timer_started_at",
            "phase",
            "current_task",
        ],
        table_rows,
    )


def print_chat_history_rows(
    conn: sqlite3.Connection,
    *,
    limit: int,
    conversation_id: str | None,
) -> None:
    columns = set(table_columns(conn, "chat_history"))
    id_column = "conversation_id" if "conversation_id" in columns else "thread_id" if "thread_id" in columns else None
    if not id_column:
        print("chat_history table not found or missing conversation_id/thread_id column")
        return

    filters: list[str] = []
    params: list[Any] = []
    if conversation_id:
        filters.append(f"{id_column} = ?")
        params.append(conversation_id)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT {id_column}, role, content, agent_key, created_at
        FROM chat_history
        {where_clause}
        ORDER BY id DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    rows = cur.fetchall()

    if not rows:
        print("chat_history: <no rows>")
        return

    table_rows: list[list[Any]] = []
    for idx, row in enumerate(rows, start=1):
        conv_id, role, content, agent_key, created_at = row
        snippet = decode_value(content)
        table_rows.append(
            [
                idx,
                decode_value(conv_id),
                decode_value(role),
                decode_value(agent_key),
                decode_value(created_at),
                snippet,
            ]
        )

    print_table(
        ["#", id_column, "role", "agent_key", "created_at", "content"],
        table_rows,
    )


def print_conversation_rows(
    conn: sqlite3.Connection,
    *,
    limit: int,
    conversation_id: str | None,
) -> None:
    if "conversations" not in list_tables(conn):
        print("conversations: <no rows>")
        return

    columns = set(table_columns(conn, "conversations"))
    if "conversation_id" not in columns:
        print("conversations: <no rows>")
        return

    filters: list[str] = []
    params: list[Any] = []
    if conversation_id:
        filters.append("conversation_id = ?")
        params.append(conversation_id)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    select_columns = [
        "conversation_id",
        "agent_key",
        "last_thread_id",
        "last_run_id",
        "last_run_status",
        "created_at",
        "updated_at",
    ]
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT {", ".join(select_columns)}
        FROM conversations
        {where_clause}
        ORDER BY updated_at DESC, conversation_id DESC
        LIMIT ?
        """,
        (*params, limit),
    )
    rows = cur.fetchall()

    if not rows:
        print("conversations: <no rows>")
        return

    table_rows: list[list[Any]] = []
    for idx, row in enumerate(rows, start=1):
        (
            conv_id,
            agent_key,
            last_thread_id,
            last_run_id,
            last_run_status,
            created_at,
            updated_at,
        ) = row
        table_rows.append(
            [
                idx,
                decode_value(conv_id),
                decode_value(agent_key),
                decode_value(last_thread_id),
                decode_value(last_run_id),
                decode_value(last_run_status),
                decode_value(created_at),
                decode_value(updated_at),
            ]
        )

    print_table(
        [
            "#",
            "conversation_id",
            "agent_key",
            "last_thread_id",
            "last_run_id",
            "last_run_status",
            "created_at",
            "updated_at",
        ],
        table_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the Solidcue SQLite database")
    parser.add_argument("--db-path", help="Path to the SQLite database")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent rows to show per table")
    parser.add_argument("--thread-id", help="Filter checkpoint rows by thread id")
    parser.add_argument("--conversation-id", help="Filter checkpoint/chat rows by conversation id")
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    print(f"DB path: {db_path}")

    if not db_path.exists():
        print("Database file does not exist")
        return 1

    with connect(db_path) as conn:
        tables = list_tables(conn)
        print_section("Tables")
        if not tables:
            print("<no tables>")
        else:
            for table_name in tables:
                print_table_summary(conn, table_name)

        print_section("Checkpoint Rows")
        print_checkpoint_rows(
            conn,
            limit=args.limit,
            thread_id=args.thread_id,
            conversation_id=args.conversation_id,
        )

        print_section("Conversations")
        print_conversation_rows(
            conn,
            limit=args.limit,
            conversation_id=args.conversation_id,
        )

        print_section("Chat History")
        print_chat_history_rows(
            conn,
            limit=args.limit,
            conversation_id=args.conversation_id,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
