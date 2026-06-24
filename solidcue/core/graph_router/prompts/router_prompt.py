from __future__ import annotations

import json
from typing import Any

from solidcue.core.graph_router.prompts.router_system_prompt import build_router_system_prompt


def _format_chat_history(chat_history: list[dict[str, str]] | None, *, limit: int = 8) -> str:
    if not isinstance(chat_history, list) or not chat_history:
        return "None"

    lines: list[str] = []
    for entry in chat_history[-limit:]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip()
        content = str(entry.get("content") or "").strip()
        if not role or not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


def _format_agents(available_agents: list[dict[str, str]]) -> str:
    if not available_agents:
        return "None"
    lines: list[str] = []
    for agent in available_agents:
        agent_key = str(agent.get("agent_key") or "").strip()
        name = str(agent.get("name") or "").strip()
        description = str(agent.get("description") or "").strip()
        if not agent_key:
            continue
        lines.append(f"- {agent_key}: {name} :: {description}")
    return "\n".join(lines) if lines else "None"


def _format_metadata(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict) or not metadata:
        return "None"
    try:
        return json.dumps(metadata, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(metadata)


def _format_retained_results(agent_results: list[dict[str, Any]] | None) -> str:
    """Summarise what earlier runs already gathered and still have in memory.

    The router uses this to tell a `reshape` (re-present retained data) from a
    `task` (fetch something new). Only entries carrying structured `data` are
    listed, since only those can actually be reshaped without re-dispatching.
    """
    if not isinstance(agent_results, list) or not agent_results:
        return "None"
    lines: list[str] = []
    for result in agent_results:
        if not isinstance(result, dict):
            continue
        if not (isinstance(result.get("data"), dict) and result.get("data")):
            continue
        agent_key = str(result.get("agent_key") or "").strip() or "unknown"
        sub_task = str(result.get("sub_task") or "").strip()
        lines.append(f"- {agent_key}: {sub_task} (structured data retained, available to reshape)")
    return "\n".join(lines) if lines else "None"


def build_router_messages(
    *,
    user_input: str,
    chat_history: list[dict[str, str]] | None,
    available_agents: list[dict[str, str]],
    metadata: dict[str, Any] | None = None,
    agent_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    runtime_context = (
        "=== RUNTIME CONTEXT ===\n"
        "AVAILABLE_AGENTS:\n"
        f"{_format_agents(available_agents)}\n\n"
        "METADATA:\n"
        f"{_format_metadata(metadata)}\n\n"
        "RETAINED_RESULTS (data already gathered earlier this session):\n"
        f"{_format_retained_results(agent_results)}\n\n"
        "CHAT_HISTORY:\n"
        f"{_format_chat_history(chat_history)}\n\n"
        "CURRENT_USER_INPUT:\n"
        f"{user_input.strip()}"
    )

    return [
        {"role": "system", "content": build_router_system_prompt()},
        {"role": "user", "content": runtime_context},
    ]
