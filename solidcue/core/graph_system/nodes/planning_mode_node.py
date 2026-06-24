"""planning_mode_node — decide whether a new agent plans statically or dynamically.

A focused LLM node (kept separate from select_tools so each prompt stays narrow).
It classifies the agent's *workflow shape*, not its topic:

- static  -> the agent runs the same fixed sequence of steps on every request,
             with only the inputs varying. Its task plan can be cached and reused.
- dynamic -> the plan shape itself varies per request (different steps, different
             counts). The plan must be rebuilt each turn and never cached.

The failure modes are asymmetric: a `dynamic` agent mislabeled `static` caches a
request-specific plan and replays it incorrectly, while a `static` agent
mislabeled `dynamic` only pays for an extra re-plan. So the prompt — and the
fallback — bias hard toward `dynamic`; `static` is emitted only on a confident
fixed-pipeline signal. Result is written to agent_spec.planning_mode.
"""

from __future__ import annotations

import logging
from typing import Any

from solidcue.core.graph_system.nodes.select_tools_node import (
    _extract_json_object,
    _get_workspace_provider,
)
from solidcue.core.graph_system.state.schema import SystemState
from solidcue.core.utils.metrics import timed_async_stream_generate

logger = logging.getLogger(__name__)

_VALID_MODES = {"static", "dynamic"}
_DEFAULT_MODE = "dynamic"


async def planning_mode_node(state: SystemState) -> dict[str, Any]:
    agent_spec = dict(state.get("agent_spec") or {})

    # Respect an explicit choice supplied upstream (e.g. a pre-built spec).
    existing = str(agent_spec.get("planning_mode") or "").strip().casefold()
    if existing in _VALID_MODES:
        agent_spec["planning_mode"] = existing
        return {"agent_spec": agent_spec}

    provider = _get_workspace_provider()
    if provider is None:
        agent_spec["planning_mode"] = _DEFAULT_MODE
        return {"agent_spec": agent_spec}

    name = str(agent_spec.get("name") or "").strip()
    description = str(agent_spec.get("description") or "").strip()
    tools = [str(t).strip() for t in (agent_spec.get("selected_tools") or []) if str(t).strip()]
    tool_lines = "\n".join(f"- {t}" for t in tools) or "(none)"

    messages = [
        {
            "role": "system",
            "content": (
                "You classify an agent's PLANNING MODE — how its execution plan is "
                "shaped, not what topic it covers.\n\n"
                "Definitions:\n"
                "- static: the agent runs the SAME fixed sequence of steps on every "
                "request; only the inputs differ (e.g. a document pipeline: fetch -> "
                "extract -> generate -> upload). Its plan can be cached and reused.\n"
                "- dynamic: the plan itself changes per request — different steps, or a "
                "different number of steps depending on what the user asks (e.g. a "
                "conversational assistant that looks up one thing or compares several). "
                "Its plan must be rebuilt each turn.\n\n"
                "Bias strongly toward 'dynamic'. Choose 'static' ONLY when the agent "
                "clearly performs one deterministic pipeline every time. When unsure, "
                "answer 'dynamic'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent name: {name}\n"
                f"Agent purpose: {description}\n"
                f"Selected tools:\n{tool_lines}\n\n"
                'Return JSON only: {"planning_mode": "static" | "dynamic"}.'
            ),
        },
    ]

    mode = _DEFAULT_MODE
    try:
        output, _metric = await timed_async_stream_generate(
            provider, messages, node_name="planning_mode"
        )
        parsed = _extract_json_object(output)
        raw = parsed.get("planning_mode") if isinstance(parsed, dict) else None
        candidate = str(raw or "").strip().casefold()
        if candidate in _VALID_MODES:
            mode = candidate
    except Exception:
        logger.exception("planning_mode_node failed; defaulting to %s", _DEFAULT_MODE)

    agent_spec["planning_mode"] = mode
    return {"agent_spec": agent_spec}
