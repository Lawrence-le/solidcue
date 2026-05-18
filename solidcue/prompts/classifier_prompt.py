from __future__ import annotations

from solidcue.prompts.classifier_system_prompt import build_classifier_system_prompt


def build_classifier_messages(
    *,
    user_input: str,
    persona: str = "",
    agent_name: str = "",
    agent_description: str = "",
) -> list[dict[str, str]]:
    system_prompt = build_classifier_system_prompt()
    runtime_context = (
        "=== RUNTIME CONTEXT ===\n"
        f"AGENT_NAME: {agent_name or 'an AI agent'}\n"
        f"AGENT_DESCRIPTION: {agent_description or ''}\n"
        "PERSONA:\n"
        f"{persona or 'No specific persona defined.'}"
    )

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": runtime_context},
        {"role": "user", "content": user_input},
    ]
