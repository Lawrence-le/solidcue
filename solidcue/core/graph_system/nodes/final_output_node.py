from __future__ import annotations

from typing import Any

from solidcue.core.graph_system.state.schema import SystemState


def final_output_node(state: SystemState) -> dict[str, Any]:
    """Finalize the system graph with a stable user-facing response."""
    response = str(state.get("final_response") or state.get("assistant_draft") or "").strip()
    if not response:
        response = "I can help with workspace setup, agent creation, or selecting an existing agent."
    return {
        "final_response": response,
        "assistant_draft": response,
    }
