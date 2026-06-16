from __future__ import annotations

import logging
from typing import Any

from solidcue.core.graph_system.state.schema import SystemState
from solidcue.services.agent_service import CreateAgentInput, write_agent_config

logger = logging.getLogger(__name__)


def _scrub_secrets(agent_spec: dict[str, Any]) -> dict[str, Any]:
    """Drop *_api_key fields once they've been written to the env store, so raw
    secrets don't linger in the checkpointed graph state."""
    return {k: v for k, v in agent_spec.items() if not str(k).endswith("_api_key")}


def write_config_node(state: SystemState) -> dict[str, Any]:
    """Build AgentConfig and write <agent_key>.yaml from agent_spec. MD files already written."""
    agent_spec = state.get("agent_spec") or {}

    try:
        input_data = CreateAgentInput(**agent_spec)
        config, path = write_agent_config(input_data)
        return {
            "created_agent_key": config.agent_key,
            "created_config_path": path,
            "agent_spec": _scrub_secrets(agent_spec),
        }
    except Exception:
        logger.exception("write_config_node failed for agent_spec=%r", agent_spec)
        msg = "Failed to write agent config — check that agent_spec has all required provider fields."
        return {
            "system_next": "final_output",
            "final_response": msg,
            "assistant_draft": msg,
        }
