"""State snapshot adapter.

A thin shaping layer over the LangGraph Server SDK (``lg_client``). It owns no
storage: threads, conversations, checkpoints and interrupts all live on the
LangGraph Server (pickle in local dev, Postgres in production). These functions
fetch raw thread state via the SDK and reshape it into the response shapes the
API/CLI expect, plus a couple of schema-only helpers off ``AgentState``.
"""
from __future__ import annotations

from typing import Any

from solidcue.core.graph_agent.state.schema import AgentState
from solidcue.services.lg_client import (
    delete_lg_thread,
    get_lg_latest_thread_id,
    get_lg_thread_by_conversation,
    get_lg_thread_interrupt,
    get_lg_thread_next_nodes,
    get_lg_thread_state,
    get_lg_thread_status,
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
    """Return the pending interrupt payload for a thread, or None.

    Delegates to the LangGraph Server, which owns the thread's checkpointed
    state. A reopened/resumable thread surfaces its interrupt here without any
    local store.
    """
    return await get_lg_thread_interrupt(thread_id)


async def load_live_state(thread_id: str) -> dict[str, Any]:
    """Return the current state values for a thread from the LangGraph Server."""
    return await get_lg_thread_state(thread_id)


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


async def get_latest_thread_id() -> str | None:
    """Return the most recently created thread id on the LangGraph Server."""
    return await get_lg_latest_thread_id()


async def get_latest_thread_id_for_conversation(conversation_id: str) -> str | None:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    if not normalized_conversation_id:
        return None
    lg_thread = await get_lg_thread_by_conversation(normalized_conversation_id)
    if not lg_thread:
        return None
    thread_id = lg_thread.get("thread_id")
    return str(thread_id) if isinstance(thread_id, str) and thread_id.strip() else None


async def delete_thread_state(thread_id: str) -> bool:
    """Delete a thread (and its checkpoints) from the LangGraph Server."""
    normalized_thread_id = thread_id.strip() if isinstance(thread_id, str) else ""
    if not normalized_thread_id:
        return False
    return await delete_lg_thread(normalized_thread_id)


async def delete_conversation_state(conversation_id: str) -> bool:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    if not normalized_conversation_id:
        return False

    lg_thread = await get_lg_thread_by_conversation(normalized_conversation_id)
    if not lg_thread:
        return False
    return await delete_lg_thread(lg_thread.get("thread_id", ""))


def load_conversation_metadata(conversation_id: str) -> dict[str, Any]:
    normalized_conversation_id = conversation_id.strip() if isinstance(conversation_id, str) else ""
    return {
        "conversation_id": normalized_conversation_id,
        "agent_key": None,
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
    # An interrupted/busy thread is resumable; so is an idle thread that was
    # cancelled mid-run — the checkpoint still has pending next nodes to
    # continue from.
    next_nodes = await get_lg_thread_next_nodes(lg_thread_id)
    resumable = thread_status in ("interrupted", "busy") or bool(next_nodes)
    return {
        "resumable": resumable,
        "next_nodes": next_nodes,
        "thread_id": lg_thread_id,
    }
