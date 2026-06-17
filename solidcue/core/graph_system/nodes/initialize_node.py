from __future__ import annotations

from typing import Any

from solidcue.core.graph_system.state.schema import SystemState
from solidcue.services.workspace_service import (
    get_agents,
    list_agent_keys,  # noqa: F401  -- re-exported for test monkeypatching
    list_system_skill_keys,
)


def initialize_node(state: SystemState) -> dict[str, Any]:
    """Populate workspace context before the system graph branches."""
    agents = get_agents()
    agent_rows = [
        {
            "agent_key": agent.agent_key,
            "name": agent.name,
            "description": agent.description,
        }
        for agent in agents
        if isinstance(agent.agent_key, str) and agent.agent_key.strip()
    ]
    agent_keys = [agent["agent_key"] for agent in agent_rows]
    return {
        "workspace_has_agents": bool(agent_rows),
        "available_agent_keys": agent_keys,
        "available_agents": agent_rows,
        "available_system_skill_keys": list_system_skill_keys(),
        "metadata": dict(state.get("metadata", {})),
        "system_intent": state.get("system_intent") or "bootstrap",
    }
