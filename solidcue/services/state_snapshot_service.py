from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from solidcue.core.graph.builder import build_agent_graph, build_async_agent_graph
from solidcue.core.state.schema import AgentState
from solidcue.services.run_engine import is_thread_resumable
from solidcue.services.chat_history_service import (
    delete_chat_history,
    delete_conversation_metadata,
    get_conversation_metadata,
    load_chat_history,
)


_EXAMPLE_STATE: dict[str, Any] = {
    "agent_key": "example_agent",
    "thread_id": "example_thread",
    "conversation_id": "example_conversation",
    "user_input": "Example user input",
    "config": {"location": "example-location"},
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
    if isinstance(state, dict):
        state = dict(state)
        conversation_id = state.get("conversation_id") if isinstance(state.get("conversation_id"), str) else None
        state["chat_history"] = load_chat_history(conversation_id or thread_id)
    return state


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


def get_latest_thread_id() -> str | None:
    db_path = resolve_checkpoint_db_path()
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
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
        conn = sqlite3.connect(str(db_path))
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

    conn = sqlite3.connect(str(db_path))
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


def delete_conversation_state(conversation_id: str) -> bool:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    if not normalized_conversation_id:
        return False

    db_path = resolve_checkpoint_db_path()
    deleted_checkpoint_rows = False
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
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
                deleted_checkpoint_rows = False
                for thread_id in thread_ids:
                    deleted_checkpoint_rows = delete_thread_state(thread_id) or deleted_checkpoint_rows
            finally:
                conn.close()
        except Exception:
            deleted_checkpoint_rows = False

    deleted_chat_rows = delete_chat_history(normalized_conversation_id)
    deleted_conversation_row = delete_conversation_metadata(normalized_conversation_id)
    return deleted_checkpoint_rows or deleted_chat_rows or deleted_conversation_row


def load_conversation_metadata(conversation_id: str) -> dict[str, Any]:
    metadata = get_conversation_metadata(conversation_id)
    if metadata:
        return metadata
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    return {
        "conversation_id": normalized_conversation_id,
        "agent_key": None,
        "worked_seconds": 0,
        "last_thread_id": None,
        "last_run_id": None,
        "last_run_status": None,
        "created_at": None,
        "updated_at": None,
    }


async def load_live_state_for_conversation(conversation_id: str) -> dict[str, Any]:
    thread_id = get_latest_thread_id_for_conversation(conversation_id)
    if not thread_id:
        return {}
    return await load_live_state(thread_id)


async def get_conversation_interrupt_payload(conversation_id: str) -> dict[str, Any] | None:
    thread_id = get_latest_thread_id_for_conversation(conversation_id)
    if not thread_id:
        return None
    return await get_thread_interrupt_payload(thread_id)


async def is_conversation_resumable(conversation_id: str) -> dict[str, Any]:
    thread_id = get_latest_thread_id_for_conversation(conversation_id)
    if not thread_id:
        return {"resumable": False, "next_nodes": [], "thread_id": None}
    payload = await is_thread_resumable(thread_id)
    payload["thread_id"] = thread_id
    return payload
