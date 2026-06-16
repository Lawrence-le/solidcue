"""select_tools_node — choose which registry tools a new agent needs.

graph_system's first LLM node. Loads the tool registry *once* (only during a
create, not on every router turn), asks the workspace provider to pick the
relevant tools for the agent's purpose, validates the choice against the
registry, and writes it into agent_spec.selected_tools. Traced via
timed_async_stream_generate, same as graph_definition's generation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from solidcue.core.graph_system.state.schema import SystemState
from solidcue.core.utils.metrics import timed_async_stream_generate

logger = logging.getLogger(__name__)


def _get_workspace_provider() -> Any:
    try:
        from solidcue.core.graph_router.nodes._shared import _PROFILE_ROUTER_PROVIDER

        return _PROFILE_ROUTER_PROVIDER
    except Exception:
        return None


def _available_tools() -> list[dict[str, str]]:
    try:
        from solidcue.tools.loader import list_tools

        tools: list[dict[str, str]] = []
        for tool in list_tools():
            tool_key = str(getattr(tool, "tool_key", "") or "").strip()
            if not tool_key:
                continue
            tools.append(
                {
                    "tool_key": tool_key,
                    "description": str(getattr(tool, "description", "") or "").strip(),
                }
            )
        return tools
    except Exception:
        return []


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def select_tools_node(state: SystemState) -> dict[str, Any]:
    agent_spec = dict(state.get("agent_spec") or {})
    available = _available_tools()
    valid_keys = {t["tool_key"] for t in available}

    # Respect tools already chosen upstream (e.g. a pre-supplied spec); only
    # auto-select when none were provided.
    existing = [
        str(t).strip()
        for t in (agent_spec.get("selected_tools") or [])
        if str(t).strip() in valid_keys
    ]
    if existing:
        agent_spec["selected_tools"] = existing
        return {"agent_spec": agent_spec}

    provider = _get_workspace_provider()
    if not available or provider is None:
        agent_spec["selected_tools"] = []
        return {"agent_spec": agent_spec}

    name = str(agent_spec.get("name") or "").strip()
    description = str(agent_spec.get("description") or "").strip()
    tool_lines = "\n".join(f"- {t['tool_key']}: {t['description']}" for t in available)
    messages = [
        {
            "role": "system",
            "content": (
                "You select the tools a new agent needs to do its job. Choose only "
                "from the provided tool list, using exact tool_keys. Pick the minimal "
                "set the agent actually needs — do not add unrelated tools."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent name: {name}\n"
                f"Agent purpose: {description}\n\n"
                f"Available tools:\n{tool_lines}\n\n"
                'Return JSON only: {"selected_tools": ["tool_key", ...]}. '
                "Use [] if none are relevant."
            ),
        },
    ]

    selected: list[str] = []
    try:
        output, _metric = await timed_async_stream_generate(
            provider, messages, node_name="select_tools"
        )
        parsed = _extract_json_object(output)
        raw = parsed.get("selected_tools") if isinstance(parsed, dict) else None
        if isinstance(raw, list):
            selected = [
                str(t).strip()
                for t in raw
                if str(t).strip() in valid_keys
            ]
    except Exception:
        logger.exception("select_tools_node failed")

    agent_spec["selected_tools"] = selected
    return {"agent_spec": agent_spec}
