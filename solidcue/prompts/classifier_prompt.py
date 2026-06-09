from __future__ import annotations

from solidcue.prompts.classifier_system_prompt import build_classifier_system_prompt


def _format_conversation_history(
    chat_history: list[dict[str, str]] | None,
    *,
    current_user_input: str,
    limit: int = 4,
) -> str:
    if not isinstance(chat_history, list) or not chat_history:
        return "None"

    normalized_current = current_user_input.strip()
    lines: list[str] = []
    for entry in chat_history[-limit:]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip()
        content = str(entry.get("content") or "").strip()
        if role != "user" or not content:
            continue
        if normalized_current and content == normalized_current:
            continue
        lines.append(f"user: {content}")
    return "\n".join(lines) if lines else "None"


def build_classifier_messages(
    *,
    user_input: str,
    persona: str = "",
    agent_name: str = "",
    agent_description: str = "",
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    system_prompt = build_classifier_system_prompt()
    runtime_context = (
        "=== RUNTIME CONTEXT ===\n"
        f"AGENT_NAME: {agent_name or 'an AI agent'}\n"
        f"AGENT_DESCRIPTION: {agent_description or ''}\n"
        "PRIOR USER CONTEXT:\n"
        f"{_format_conversation_history(chat_history, current_user_input=user_input)}\n"
        "PERSONA:\n"
        f"{persona or 'No specific persona defined.'}"
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": runtime_context},
        {"role": "user", "content": user_input},
    ]
