from __future__ import annotations

from typing import Any

from solidcue.core.graph_router.nodes._shared import normalize_text
from solidcue.core.graph_router.state.schema import RouterState


def final_output_node(state: RouterState) -> dict[str, Any]:
    final_response = normalize_text(state.get("final_response")) or normalize_text(
        state.get("assistant_draft")
    )
    updates: dict[str, Any] = {"final_response": final_response}

    # Append the assistant's reply to the persisted chat_history channel (operator.add).
    # Paired with the user-message write in initialize_router_node, this keeps the full
    # conversation history in graph state so turn N+1 reads it without a side-DB call.
    if final_response:
        updates["chat_history"] = [{"role": "assistant", "content": final_response}]

    return updates
