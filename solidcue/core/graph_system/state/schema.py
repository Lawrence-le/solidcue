from __future__ import annotations

from typing import Any, Annotated, Literal, TypedDict
import operator


class SystemState(TypedDict, total=False):
    """State for the no-agent-key system graph."""

    thread_id: str
    conversation_id: str
    user_input: str
    metadata: dict[str, Any]

    system_intent: Literal[
        "bootstrap",
        "setup_provider",
        "create_agent",
        "import_agent",
        "select_agent",
        "repair_config",
        "final_output",
    ]
    system_next: Literal["final_output"]
    route_reason: str
    assistant_draft: str
    final_response: str
    system_skill_key: str
    system_skill_path: str
    system_skill: str

    workspace_has_agents: bool
    available_agent_keys: list[str]
    available_agents: Annotated[list[dict[str, Any]], operator.add]
    messages: Annotated[list[dict[str, Any]], operator.add]
    available_system_skill_keys: list[str]
