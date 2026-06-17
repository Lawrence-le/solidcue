from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from solidcue.core.graph_agent.builder import build_async_agent_graph
from solidcue.core.graph_agent.state.schema import AgentState
from solidcue.services.lg_client import (
    delete_lg_thread,
    get_lg_thread_by_conversation,
    get_lg_thread_state,
    get_lg_thread_status,
)


_EXAMPLE_STATE: dict[str, Any] = {
    "agent_key": "example_agent",
    "thread_id": "example_thread",
    "conversation_id": "example_conversation",
    "user_input": "Example user input",
    "config": {"location": "example-location"},
    "worked_seconds": 0,
    "timer_started_at": None,
    "metadata": {
        "source_paths": ["example/source"],
        "output_paths": ["example/output"],
        "source_filenames": ["source.txt"],
        "output_filenames": ["result.txt"],
        "current_task_id": "task_1",
        "total_tasks": 1,
    },
    "messages": [],
    "chat_history": [],
    "llm_prompt_messages": [],
    "max_retries": 3,
    "phase": "decision",
    "synthesis_draft": "",
    "failure_type": "example_failure",
    "validation_report": {},
    "final_response": "",
    "task_plan": [
        {
            "id": "task_1",
            "type": "example_task",
            "description": "Example task description",
            "requires": ["example_dependency"],
            "context": {"source_path": "example/source", "source_filename": "source.txt"},
            "status": "pending",
        }
    ],
    "current_task": "task_1",
    "router_next": "decision",
    "source_attempt": 0,
    "artifact_attempt": 0,
    "synthesis_attempt": 0,
    "active_tool_call": {
        "action": "use_tool",
        "thought": "Example tool invocation",
        "tool_name": "example_tool",
        "tool_input": {"id": "example-id"},
        "approval_preview": None,
    },
    "decision": {
        "action": "use_tool",
        "thought": "Example tool invocation",
        "tool_name": "example_tool",
        "tool_input": {"id": "example-id"},
        "approval_preview": None,
    },
    "tool_use": True,
    "tool_call_history": [
        {
            "task_id": "task_1",
            "tool_name": "example_tool",
            "tool_input": {"id": "example-id"},
            "success": False,
            "output": None,
            "execution_result": {
                "success": False,
                "type": "tool_execution",
                "content": None,
                "error": "Example error",
            },
        }
    ],
    "tool_turn_count": 1,
    "execution_result": {
        "success": False,
        "type": "tool_execution",
        "content": None,
        "error": "Example error",
    },
    "handoff": {},
    "retry_reason": "RETRY_STATUS: EXAMPLE",
}


def list_agent_state_keys() -> list[str]:
    return sorted(AgentState.__annotations__.keys())


def build_state_snapshot(*, keys: list[str] | None = None, include_all: bool = False) -> dict[str, Any]:
    schema_keys = set(AgentState.__annotations__.keys())
    if include_all or not keys:
        return {key: _EXAMPLE_STATE.get(key) for key in sorted(schema_keys)}

    selected = [key for key in keys if key in schema_keys]
    return {key: _EXAMPLE_STATE.get(key) for key in selected}


async def get_thread_interrupt_payload(thread_id: str) -> dict[str, Any] | None:
    """Return the pending interrupt payload from the checkpoint, or None if no interrupt is pending."""
    graph = await build_async_agent_graph()
    try:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return None
    if not snapshot:
        return None
    for task in getattr(snapshot, "tasks", None) or []:
        for interrupt in getattr(task, "interrupts", None) or []:
            value = getattr(interrupt, "value", None)
            if isinstance(value, dict):
                return value
    return None


async def load_live_state(thread_id: str) -> dict[str, Any]:
    graph = await build_async_agent_graph()
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = getattr(snapshot, "values", None)
    state = values if isinstance(values, dict) else {}
    return dict(state) if isinstance(state, dict) else {}


async def build_live_state_snapshot(
    *,
    thread_id: str,
    keys: list[str] | None = None,
    include_all: bool = False,
) -> dict[str, Any]:
    live_state = await load_live_state(thread_id)
    schema_keys = set(AgentState.__annotations__.keys())
    if include_all or not keys:
        return {key: live_state.get(key) for key in sorted(schema_keys)}
    selected = [key for key in keys if key in schema_keys]
    return {key: live_state.get(key) for key in selected}


def resolve_checkpoint_db_path() -> Path:
    configured_path = os.getenv("SOLIDCUE_CHECKPOINT_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".solidcue" / "checkpoints.sqlite"


def _connect_checkpoint_db() -> sqlite3.Connection:
    db_path = resolve_checkpoint_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=1.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout = 1000")
    # Request incremental auto-vacuum so free pages can be reclaimed without a
    # full rewrite. On an existing NONE database this only takes effect after the
    # next VACUUM (see reclaim_checkpoint_db_space); on a fresh file it applies
    # immediately, before any tables are created.
    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
    return conn


def get_latest_thread_id() -> str | None:
    db_path = resolve_checkpoint_db_path()
    if not db_path.exists():
        return None
    try:
        conn = _connect_checkpoint_db()
        cur = conn.cursor()
        cur.execute("SELECT thread_id FROM checkpoints ORDER BY rowid DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        thread_id = row[0]
        return str(thread_id) if isinstance(thread_id, str) and thread_id.strip() else None
    except Exception:
        return None


def get_latest_thread_id_for_conversation(conversation_id: str) -> str | None:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    if not normalized_conversation_id:
        return None
    db_path = resolve_checkpoint_db_path()
    if not db_path.exists():
        return None
    try:
        conn = _connect_checkpoint_db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT thread_id
            FROM checkpoints
            WHERE checkpoint_ns = ''
              AND json_extract(metadata, '$.conversation_id') = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (normalized_conversation_id,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        thread_id = row[0]
        return str(thread_id) if isinstance(thread_id, str) and thread_id.strip() else None
    except Exception:
        return None


def delete_thread_state(thread_id: str) -> bool:
    db_path = resolve_checkpoint_db_path()
    if not db_path.exists():
        return False

    conn = _connect_checkpoint_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        table_rows = cur.fetchall()

        deleted_any = False
        for row in table_rows:
            table_name = row[0]
            if not isinstance(table_name, str) or not table_name:
                continue
            quoted_table_name = f'"{table_name}"'
            cur.execute(f"PRAGMA table_info({quoted_table_name})")
            columns = cur.fetchall()
            has_thread_id = any(column[1] == "thread_id" for column in columns if len(column) > 1)
            if not has_thread_id:
                continue
            cur.execute(f"DELETE FROM {quoted_table_name} WHERE thread_id = ?", (thread_id,))
            if cur.rowcount > 0:
                deleted_any = True

        if deleted_any:
            conn.commit()
        else:
            conn.rollback()
        return deleted_any
    finally:
        conn.close()


def _resolve_checkpoint_keep_last() -> int:
    """How many of the most recent checkpoints to retain per thread when pruning.

    Resuming an interrupted thread only needs the latest checkpoint, so the
    default keeps a small buffer for safety. Override with
    ``SOLIDCUE_CHECKPOINT_KEEP_LAST``.
    """
    raw = os.getenv("SOLIDCUE_CHECKPOINT_KEEP_LAST")
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError:
        return 1
    return value if value >= 1 else 1


def prune_thread_checkpoints(thread_id: str, *, keep_last: int | None = None) -> int:
    """Delete all but the most recent ``keep_last`` checkpoints for a thread.

    LangGraph retains the full checkpoint history per thread by default; for a
    finished thread the older steps are dead weight. Keeps the newest
    ``keep_last`` checkpoints per ``checkpoint_ns`` and removes any writes rows
    that no longer reference a surviving checkpoint. Returns the number of
    checkpoint rows deleted.
    """
    normalized_thread_id = thread_id.strip() if isinstance(thread_id, str) else ""
    if not normalized_thread_id:
        return 0

    retain = keep_last if isinstance(keep_last, int) and keep_last >= 1 else _resolve_checkpoint_keep_last()

    db_path = resolve_checkpoint_db_path()
    if not db_path.exists():
        return 0

    conn = _connect_checkpoint_db()
    try:
        cur = conn.cursor()
        # checkpoint_id is monotonic (sortable UUID); newest sort last.
        cur.execute(
            """
            DELETE FROM checkpoints
            WHERE thread_id = ?
              AND rowid IN (
                SELECT rowid FROM (
                    SELECT rowid,
                           ROW_NUMBER() OVER (
                               PARTITION BY thread_id, checkpoint_ns
                               ORDER BY checkpoint_id DESC
                           ) AS rn
                    FROM checkpoints
                    WHERE thread_id = ?
                ) WHERE rn > ?
              )
            """,
            (normalized_thread_id, normalized_thread_id, retain),
        )
        deleted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

        # Drop writes orphaned by the checkpoint deletions above.
        cur.execute(
            """
            DELETE FROM writes
            WHERE thread_id = ?
              AND checkpoint_id NOT IN (
                SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?
              )
            """,
            (normalized_thread_id, normalized_thread_id),
        )

        if deleted or (cur.rowcount and cur.rowcount > 0):
            conn.commit()
        else:
            conn.rollback()
        return deleted
    finally:
        conn.close()


def reclaim_checkpoint_db_space(*, full: bool = False) -> bool:
    """Reclaim free pages left behind by checkpoint/writes deletes.

    On a database that still has ``auto_vacuum=NONE`` (the historical default),
    a one-time full ``VACUUM`` converts it to incremental auto-vacuum and frees
    every dead page at once. Afterwards (or when ``full`` is False) the much
    cheaper ``PRAGMA incremental_vacuum`` releases pages on the free list
    without rewriting the whole file. Returns True if any reclaim ran.
    """
    db_path = resolve_checkpoint_db_path()
    if not db_path.exists():
        return False

    # VACUUM cannot run inside a transaction — use autocommit mode.
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        auto_vacuum = conn.execute("PRAGMA auto_vacuum").fetchone()
        mode = auto_vacuum[0] if auto_vacuum else 0
        # mode: 0=NONE, 1=FULL, 2=INCREMENTAL
        if full or mode != 2:
            # Switching the auto_vacuum mode only takes effect on a full VACUUM.
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.execute("VACUUM")
        else:
            conn.execute("PRAGMA incremental_vacuum")
        return True
    except sqlite3.OperationalError:
        # Database busy / locked — skip; maintenance will retry on a later run.
        return False
    finally:
        conn.close()


def run_checkpoint_maintenance(
    thread_id: str,
    *,
    keep_last: int | None = None,
    reclaim: bool = True,
) -> dict[str, Any]:
    """Prune a finished thread's checkpoint history and reclaim free space.

    Safe to call after a run reaches a terminal state. Failures are swallowed so
    maintenance never breaks the run path. Returns a small summary dict.
    """
    pruned = 0
    reclaimed = False
    try:
        pruned = prune_thread_checkpoints(thread_id, keep_last=keep_last)
    except Exception:
        pruned = 0
    if reclaim:
        try:
            reclaimed = reclaim_checkpoint_db_space()
        except Exception:
            reclaimed = False
    return {"thread_id": thread_id, "pruned": pruned, "reclaimed": reclaimed}


async def delete_conversation_state(conversation_id: str) -> bool:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    if not normalized_conversation_id:
        return False

    deleted_checkpoint_rows = False
    db_path = resolve_checkpoint_db_path()
    if db_path.exists():
        try:
            conn = _connect_checkpoint_db()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT DISTINCT thread_id
                    FROM checkpoints
                    WHERE checkpoint_ns = ''
                      AND json_extract(metadata, '$.conversation_id') = ?
                    """,
                    (normalized_conversation_id,),
                )
                thread_ids = [str(row[0]) for row in cur.fetchall() if row and isinstance(row[0], str) and row[0].strip()]
                for thread_id in thread_ids:
                    deleted_checkpoint_rows = delete_thread_state(thread_id) or deleted_checkpoint_rows
            finally:
                conn.close()
        except Exception:
            deleted_checkpoint_rows = False

    # Also delete the LangGraph Server thread (best-effort).
    try:
        lg_thread = await get_lg_thread_by_conversation(normalized_conversation_id)
        if lg_thread:
            deleted_lg = await delete_lg_thread(lg_thread.get("thread_id", ""))
            deleted_checkpoint_rows = deleted_checkpoint_rows or deleted_lg
    except Exception:
        pass

    return deleted_checkpoint_rows


def load_conversation_metadata(conversation_id: str) -> dict[str, Any]:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    return {
        "conversation_id": normalized_conversation_id,
        "agent_key": None,
        "worked_seconds": 0,
        "timer_started_at": None,
        "last_thread_id": None,
        "last_run_id": None,
        "last_run_status": None,
        "created_at": None,
        "updated_at": None,
    }


async def load_conversation_snapshot(conversation_id: str) -> dict[str, Any]:
    """Load conversation snapshot from LangGraph Server thread state."""
    normalized = conversation_id.strip() if isinstance(conversation_id, str) else ""
    lg_thread = await get_lg_thread_by_conversation(normalized)
    if lg_thread:
        lg_thread_id = lg_thread.get("thread_id", "")
        values = await get_lg_thread_state(lg_thread_id)
        return {
            "conversation_id": normalized,
            "agent_key": values.get("target_agent_key") or values.get("agent_key") or None,
            "worked_seconds": int(values.get("worked_seconds") or 0),
            "timer_started_at": None,
            "last_thread_id": lg_thread_id,
            "last_run_id": None,
            "last_run_status": None,
            "created_at": None,
            "updated_at": None,
            "chat_history": values.get("chat_history") or [],
        }
    # No LG Server thread found — return empty snapshot.
    return {
        "conversation_id": normalized,
        "agent_key": None,
        "worked_seconds": 0,
        "timer_started_at": None,
        "last_thread_id": None,
        "last_run_id": None,
        "last_run_status": None,
        "created_at": None,
        "updated_at": None,
        "chat_history": [],
    }


async def load_live_state_for_conversation(conversation_id: str) -> dict[str, Any]:
    """Load live state values from LangGraph Server thread."""
    normalized = conversation_id.strip() if isinstance(conversation_id, str) else ""
    lg_thread = await get_lg_thread_by_conversation(normalized)
    if lg_thread:
        return await get_lg_thread_state(lg_thread.get("thread_id", ""))
    return {}


async def get_conversation_interrupt_payload(conversation_id: str) -> dict[str, Any] | None:
    """Return interrupt payload from LangGraph Server thread state, or None."""
    normalized = conversation_id.strip() if isinstance(conversation_id, str) else ""
    lg_thread = await get_lg_thread_by_conversation(normalized)
    if not lg_thread:
        return None
    return await get_thread_interrupt_payload(lg_thread.get("thread_id", ""))


async def is_conversation_resumable(conversation_id: str) -> dict[str, Any]:
    """Check LangGraph Server thread status to determine resumability."""
    normalized = conversation_id.strip() if isinstance(conversation_id, str) else ""
    lg_thread = await get_lg_thread_by_conversation(normalized)
    if not lg_thread:
        return {"resumable": False, "next_nodes": [], "thread_id": None}
    lg_thread_id = lg_thread.get("thread_id", "")
    thread_status = await get_lg_thread_status(lg_thread_id)
    resumable = thread_status in ("interrupted", "busy")
    return {
        "resumable": resumable,
        "next_nodes": [],
        "thread_id": lg_thread_id,
    }
