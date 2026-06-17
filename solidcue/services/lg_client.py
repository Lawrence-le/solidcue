"""LangGraph Server SDK client — singleton for backend proxy use.

All state/thread/run queries that previously read from the local checkpoint
SQLite or run_engine in-memory dicts now go through this client against the
LangGraph Server (default: http://localhost:2024).
"""
from __future__ import annotations

import os
from typing import Any

from langgraph_sdk import get_client as _sdk_get_client

_LANGGRAPH_URL = os.environ.get("LANGGRAPH_API_URL", "http://localhost:2024")

_client: Any = None


def get_lg_client() -> Any:
    global _client
    if _client is None:
        _client = _sdk_get_client(url=_LANGGRAPH_URL)
    return _client


async def get_lg_thread_by_conversation(conversation_id: str) -> dict[str, Any] | None:
    """Return the first LangGraph Server thread whose metadata.conversation_id matches."""
    try:
        client = get_lg_client()
        results = await client.threads.search(
            metadata={"conversation_id": conversation_id},
            limit=1,
        )
        return results[0] if results else None
    except Exception:
        return None


async def get_lg_thread_state(lg_thread_id: str) -> dict[str, Any]:
    """Return the current state values for a thread (empty dict on error)."""
    try:
        client = get_lg_client()
        snapshot = await client.threads.get_state(lg_thread_id)
        values = snapshot.get("values") if isinstance(snapshot, dict) else getattr(snapshot, "values", None)
        return values if isinstance(values, dict) else {}
    except Exception:
        return {}


async def get_lg_thread_status(lg_thread_id: str) -> str:
    """Return the thread status: idle | busy | interrupted | error."""
    try:
        client = get_lg_client()
        thread = await client.threads.get(lg_thread_id)
        if isinstance(thread, dict):
            return str(thread.get("status") or "idle")
        return str(getattr(thread, "status", "idle"))
    except Exception:
        return "idle"


async def delete_lg_thread(lg_thread_id: str) -> bool:
    """Delete a thread from the LangGraph Server. Returns True on success."""
    try:
        client = get_lg_client()
        await client.threads.delete(lg_thread_id)
        return True
    except Exception:
        return False


async def get_lg_thread_interrupt(lg_thread_id: str) -> dict[str, Any] | None:
    """Return the pending interrupt payload for a thread, or None.

    Reads the thread's current state snapshot from the LangGraph Server and walks
    its pending tasks for an interrupt value. The server owns this state, so a
    reopened/resumable thread surfaces its interrupt here without any local store.
    """
    try:
        client = get_lg_client()
        snapshot = await client.threads.get_state(lg_thread_id)
    except Exception:
        return None
    if not snapshot:
        return None
    tasks = snapshot.get("tasks") if isinstance(snapshot, dict) else getattr(snapshot, "tasks", None)
    for task in tasks or []:
        interrupts = task.get("interrupts") if isinstance(task, dict) else getattr(task, "interrupts", None)
        for interrupt in interrupts or []:
            value = interrupt.get("value") if isinstance(interrupt, dict) else getattr(interrupt, "value", None)
            if isinstance(value, dict):
                return value
    return None


async def get_lg_latest_thread_id() -> str | None:
    """Return the most recently created thread id on the server, or None."""
    try:
        client = get_lg_client()
        results = await client.threads.search(limit=1, sort_by="created_at", sort_order="desc")
    except Exception:
        return None
    if not results:
        return None
    thread = results[0]
    thread_id = thread.get("thread_id") if isinstance(thread, dict) else getattr(thread, "thread_id", None)
    return str(thread_id) if isinstance(thread_id, str) and thread_id.strip() else None
