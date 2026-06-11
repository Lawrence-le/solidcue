from __future__ import annotations

from typing import Any

from solidcue.core.graph_router.nodes._shared import normalize_text
from solidcue.core.graph_router.state.schema import RouterState
from solidcue.services.chat_history_service import load_chat_history


def initialize_router_node(state: RouterState) -> dict[str, Any]:
    conversation_id = normalize_text(state.get("conversation_id")) or normalize_text(
        state.get("thread_id")
    )
    updates: dict[str, Any] = {}
    if conversation_id:
        updates["conversation_id"] = conversation_id
        if not state.get("chat_history"):
            updates["chat_history"] = load_chat_history(conversation_id, limit=8)
    updates["worked_seconds"] = int(state.get("worked_seconds") or 0)
    updates["timer_started_at"] = state.get("timer_started_at")
    return updates
