from __future__ import annotations

from solidcue.core.graph_system.state.schema import SystemState
from solidcue.services.workspace_service import get_system_skill_path, load_system_skill, resolve_system_skill_key


def _looks_like_setup_request(user_input: str) -> bool:
    lowered = user_input.casefold()
    return any(
        keyword in lowered
        for keyword in (
            "provider",
            "api key",
            "api-key",
            "setup",
            "configure",
            "onboard",
            "bootstrap",
            "repair",
            "fix",
            "import",
            "clone",
        )
    )


def intent_node(state: SystemState) -> dict[str, object]:
    """Classify the no-agent-key workflow into a bootstrap-oriented intent."""
    user_input = str(state.get("user_input") or "").strip()
    if not user_input:
        system_intent = "bootstrap"
        skill_key = resolve_system_skill_key(system_intent)
        return {
            "system_intent": system_intent,
            "system_next": "final_output",
            "route_reason": "No user input provided for system workflow.",
            "assistant_draft": "I need a little more detail before I can help set up the workspace.",
            "final_response": "I need a little more detail before I can help set up the workspace.",
            "system_skill_key": skill_key,
            "system_skill_path": str(get_system_skill_path(system_intent)),
            "system_skill": load_system_skill(system_intent),
        }

    workspace_has_agents = bool(state.get("workspace_has_agents"))

    if _looks_like_setup_request(user_input):
        if "provider" in user_input.casefold() or "api key" in user_input.casefold():
            intent = "setup_provider"
            message = "I can help set up providers and workspace configuration."
        elif "import" in user_input.casefold() or "clone" in user_input.casefold():
            intent = "import_agent"
            message = "I can help import or clone an agent template."
        elif "repair" in user_input.casefold() or "fix" in user_input.casefold():
            intent = "repair_config"
            message = "I can help repair workspace configuration."
        else:
            intent = "create_agent"
            message = "I can help create a new agent from scratch."
        skill_key = resolve_system_skill_key(intent)
        return {
            "system_intent": intent,
            "system_next": "final_output",
            "route_reason": "User asked for a workspace setup action.",
            "assistant_draft": message,
            "final_response": message,
            "system_skill_key": skill_key,
            "system_skill_path": str(get_system_skill_path(intent)),
            "system_skill": load_system_skill(intent),
        }

    if workspace_has_agents:
        message = "You already have agents configured. Pick one to continue, or ask me to create a new agent."
        intent = "select_agent"
        skill_key = resolve_system_skill_key(intent)
        return {
            "system_intent": intent,
            "system_next": "final_output",
            "route_reason": "Workspace already has agents available.",
            "assistant_draft": message,
            "final_response": message,
            "system_skill_key": skill_key,
            "system_skill_path": str(get_system_skill_path(intent)),
            "system_skill": load_system_skill(intent),
        }

    message = "No agents are configured yet. Create an agent or set up the workspace first."
    intent = "create_agent"
    skill_key = resolve_system_skill_key(intent)
    return {
        "system_intent": intent,
        "system_next": "final_output",
        "route_reason": "Workspace has no runnable agents yet.",
        "assistant_draft": message,
        "final_response": message,
        "system_skill_key": skill_key,
        "system_skill_path": str(get_system_skill_path(intent)),
        "system_skill": load_system_skill(intent),
    }
