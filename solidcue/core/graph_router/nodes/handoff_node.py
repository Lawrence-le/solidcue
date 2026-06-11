from __future__ import annotations

from solidcue.core.graph_router.nodes._shared import normalize_text
from solidcue.core.graph_router.state.schema import RouterState


def handoff_node(state: RouterState) -> dict[str, str]:
    handoff = state.get("handoff")
    if not isinstance(handoff, dict):
        existing_response = normalize_text(state.get("final_response")) or normalize_text(
            state.get("assistant_draft")
        )
        return {"router_next": "final_output", "final_response": existing_response}
    action = handoff.get("action")
    if action == "create_agent":
        existing_response = normalize_text(state.get("final_response")) or normalize_text(
            state.get("assistant_draft")
        )
        return {
            "router_next": "final_output",
            "final_response": existing_response
            or "I can help create a new agent. Tell me the agent name, description, and tools.",
        }
    existing_response = normalize_text(state.get("final_response")) or normalize_text(
        state.get("assistant_draft")
    )
    return {
        "router_next": "final_output",
        "final_response": existing_response
        or normalize_text(handoff.get("task_input")),
    }
