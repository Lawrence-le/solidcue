from __future__ import annotations

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


def build_router_messages(
    *,
    user_input: str,
    chat_history: list[dict[str, str]] | None,
    available_agents: list[dict[str, str]],
) -> list[dict[str, str]]:
    runtime_context = (
        "=== RUNTIME CONTEXT ===\n"
        "AVAILABLE_AGENTS:\n"
        f"{_format_agents(available_agents)}\n\n"
        "CHAT_HISTORY:\n"
        f"{_format_chat_history(chat_history)}\n\n"
        "CURRENT_USER_INPUT:\n"
        f"{user_input.strip()}"
    )

    return [
        {"role": "system", "content": build_router_system_prompt()},
        {"role": "user", "content": runtime_context},
    ]
