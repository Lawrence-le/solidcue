from __future__ import annotations

import json
from typing import Any


def _build_plan_system_prompt() -> str:
    return """
You are the planning layer for a multi-agent workspace. The request has already been
classified as work that needs one or more agents. Your only job is to write the
execution plan: an ordered list of steps, each naming an agent and the sub-task it
should perform.

How to plan:
- Use ONLY agent_keys listed in AVAILABLE_AGENTS. Never invent an agent_key.
- Each step is { "agent_key": ..., "sub_task": ... }. Write each sub_task as a clear,
  self-contained instruction for that agent.
- A sub_task must only describe what THAT agent should do with its own tools. Do NOT
  ask an agent to combine its result with data it cannot fetch itself — combining
  results across steps is handled later, not by the agent.
- If the request names multiple distinct subjects that each need their own retrieval,
  create one step per subject.
- Order steps so that any step depending on an earlier step's output comes after it.
- RETAINED_RESULTS lists data already gathered this session. Do NOT re-plan steps to
  fetch data that is already there — only plan steps for what is genuinely missing.
- target_artifacts_source: if the user supplied source references (URLs, file ids) the
  work should act on, list them; otherwise return an empty list.

Return JSON only, no preamble:
{
  "plan": [ { "agent_key": "agent_key", "sub_task": "what this agent should do" } ],
  "target_artifacts_source": [
    { "index": 1, "source_type": "url", "source_ref": "https://...", "item_key": "u_abc" }
  ]
}
""".strip()


def _format_agents(available_agents: list[dict[str, str]]) -> str:
    if not available_agents:
        return "None"
    lines: list[str] = []
    for agent in available_agents:
        agent_key = str(agent.get("agent_key") or "").strip()
        if not agent_key:
            continue
        name = str(agent.get("name") or "").strip()
        description = str(agent.get("description") or "").strip()
        lines.append(f"- {agent_key}: {name} :: {description}")
    return "\n".join(lines) if lines else "None"


def _format_chat_history(chat_history: list[dict[str, str]] | None, *, limit: int = 6) -> str:
    if not isinstance(chat_history, list) or not chat_history:
        return "None"
    lines: list[str] = []
    for entry in chat_history[-limit:]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip()
        content = str(entry.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


def _format_retained_results(agent_results: list[dict[str, Any]] | None) -> str:
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
        lines.append(f"- {agent_key}: {sub_task} (data already retained)")
    return "\n".join(lines) if lines else "None"


def _format_metadata(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict) or not metadata:
        return "None"
    try:
        return json.dumps(metadata, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(metadata)


def build_plan_messages(
    *,
    user_input: str,
    available_agents: list[dict[str, str]],
    chat_history: list[dict[str, str]] | None = None,
    agent_results: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    runtime_context = (
        "=== AVAILABLE_AGENTS ===\n"
        f"{_format_agents(available_agents)}\n\n"
        "=== METADATA ===\n"
        f"{_format_metadata(metadata)}\n\n"
        "=== RETAINED_RESULTS (already gathered — do not re-fetch) ===\n"
        f"{_format_retained_results(agent_results)}\n\n"
        "=== CHAT_HISTORY ===\n"
        f"{_format_chat_history(chat_history)}\n\n"
        "=== TASK TO PLAN ===\n"
        f"{(user_input or '').strip()}\n\n"
        "Write the execution plan now."
    )
    return [
        {"role": "system", "content": _build_plan_system_prompt()},
        {"role": "user", "content": runtime_context},
    ]
