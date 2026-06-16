"""generate_definitions_node — generate PERSONA / SKILL / TOOLS for a new agent.

Runs the three graph_definition subgraphs sequentially inside ONE Langfuse span,
so create-agent generation shows up as a single cohesive trace (mirroring how
graph_agent's ``solidcue:agent:<key>`` is one trace) rather than three fragments.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_config as _get_config

from solidcue.core.graph_system.state.schema import SystemState
from solidcue.observability.langfuse import start_langfuse_span

logger = logging.getLogger(__name__)

_definition_graph_cache: dict[str, Any] = {}
_TARGETS = ("persona", "skill", "tools")


def _get_definition_graph(target: str) -> Any:
    if target not in _definition_graph_cache:
        from solidcue.core.graph_definition.builder import build_definition_graph
        _definition_graph_cache[target] = build_definition_graph(target)
    return _definition_graph_cache[target]


def _parent_config() -> dict[str, Any]:
    """Carry the parent run's Langfuse callbacks + metadata into the subgraph runs."""
    config: dict[str, Any] = {}
    try:
        parent = _get_config()
        callbacks = parent.get("callbacks")
        metadata = parent.get("metadata")
        if callbacks is not None:
            config["callbacks"] = callbacks
        if metadata is not None:
            config["metadata"] = dict(metadata)
    except Exception:
        return {}
    return config


async def _run_one(
    target: str, agent_key: str, agent_spec: dict[str, Any], config: dict[str, Any]
) -> tuple[str, str]:
    subgraph_input: dict[str, Any] = {
        "definition_target": target,
        "agent_key": agent_key,
        "agent_spec": agent_spec,
        "overwrite": True,
    }
    content = ""
    path = ""
    graph = _get_definition_graph(target)
    async for mode, chunk in graph.astream(
        subgraph_input, config=config or None, stream_mode=["updates"]
    ):
        if mode == "updates" and isinstance(chunk, dict):
            write_update = chunk.get("write")
            if isinstance(write_update, dict):
                path = str(write_update.get("definition_path") or "")
            gen_update = chunk.get("generate")
            if isinstance(gen_update, dict):
                content = str(gen_update.get("definition_content") or "")
    return content, path


async def generate_definitions_node(state: SystemState) -> dict[str, Any]:
    """Generate all three definition files under one create-agent span."""
    agent_key = str(
        state.get("created_agent_key") or state.get("agent_spec", {}).get("agent_key", "")
    ).strip()
    agent_spec = state.get("agent_spec") or {}
    config = _parent_config()

    artifacts: list[dict[str, Any]] = []
    # One span wraps all three generations → a single cohesive create-agent trace.
    with start_langfuse_span(
        name=f"solidcue:create_agent:{agent_key}",
        input_payload={"agent_key": agent_key},
        metadata={"agent_key": agent_key},
    ):
        for target in _TARGETS:
            try:
                content, path = await _run_one(target, agent_key, agent_spec, config)
            except Exception:
                logger.exception("generate_definitions: %s failed", target)
                content, path = "", ""
            artifacts.append({"target": target, "path": path, "content": content})

    return {"artifacts": artifacts}
