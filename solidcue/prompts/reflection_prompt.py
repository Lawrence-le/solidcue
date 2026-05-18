from __future__ import annotations

from typing import Any

from solidcue.prompts.reflection_system_prompt import build_reflection_system_prompt


def build_reflection_messages(
    *,
    tool_name: str,
    requires: list[str],
    execution_result: Any,
) -> list[dict[str, str]]:
    system_prompt = build_reflection_system_prompt()
    requires_text = ", ".join(f'"{r}"' for r in requires if isinstance(r, str) and r.strip())
    runtime_context = (
        f"TOOL_NAME: {tool_name or 'unknown'}\n"
        f"REQUIREMENTS: {requires_text}\n\n"
        "EXECUTION_RESULT:\n"
        f"{execution_result}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": runtime_context},
    ]

