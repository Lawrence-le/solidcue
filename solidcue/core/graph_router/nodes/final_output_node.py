from __future__ import annotations

from solidcue.core.graph_router.nodes._shared import normalize_text
from solidcue.core.graph_router.state.schema import RouterState


def final_output_node(state: RouterState) -> dict[str, str]:
    final_response = normalize_text(state.get("final_response"))
    if final_response:
        return {"final_response": final_response}
    assistant_draft = normalize_text(state.get("assistant_draft"))
    return {"final_response": assistant_draft}
