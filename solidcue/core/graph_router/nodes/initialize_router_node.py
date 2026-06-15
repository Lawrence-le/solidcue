from __future__ import annotations

from typing import Any

from solidcue.core.graph_router.nodes._shared import normalize_text
from solidcue.core.graph_router.state.schema import RouterState


def initialize_router_node(state: RouterState) -> dict[str, Any]:
    conversation_id = normalize_text(state.get("conversation_id")) or normalize_text(
        state.get("thread_id")
    )
    updates: dict[str, Any] = {}
    if conversation_id:
        updates["conversation_id"] = conversation_id
    updates["worked_seconds"] = int(state.get("worked_seconds") or 0)
    updates["timer_started_at"] = state.get("timer_started_at")

    # Append the user's turn to the persisted chat_history channel (operator.add).
    # Under LangGraph Server the thread checkpoint accumulates history across turns;
    # under the FastAPI + SqliteSaver path the same accumulation happens via the
    # graph checkpoint.  Either way, reading state["chat_history"] in downstream
    # nodes gives the full conversation history for this thread.
    user_input = normalize_text(state.get("user_input"))
    if user_input:
        updates["chat_history"] = [{"role": "user", "content": user_input}]

    return updates
