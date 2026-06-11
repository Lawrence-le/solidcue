from __future__ import annotations

from solidcue.core.router_graph.router_node._shared import normalize_text
from solidcue.core.router_graph.state import RouterState


def handoff_node(state: RouterState) -> dict[str, str]:
    handoff = state.get("handoff")
    if not isinstance(handoff, dict):
        return {"router_next": "final_output", "final_response": ""}
    action = handoff.get("action")
    if action == "create_agent":
        return {
            "router_next": "final_output",
            "final_response": "I can help create a new agent. Tell me the agent name, description, and tools.",
        }
    return {
        "router_next": "final_output",
        "final_response": normalize_text(handoff.get("task_input")) or normalize_text(
            state.get("assistant_draft")
        ),
    }
