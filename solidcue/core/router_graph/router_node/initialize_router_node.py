from __future__ import annotations

from typing import Any

from solidcue.core.router_graph.router_node._shared import normalize_text
from solidcue.core.router_graph.state import RouterState
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
    return updates
