import json
from typing import Any

from solidcue.prompts.reflection_system_prompt import build_reflection_system_prompt


def build_reflection_messages(
    *,
    user_query: str,
    execution_result: dict[str, Any] | None,
    retry_reason: str | None,
    tool_stage: str | None = None,
) -> list[dict[str, str]]:
    execution_json = json.dumps(execution_result or {}, ensure_ascii=True, default=str)

    system_prompt = build_reflection_system_prompt()

    user_content = (
        f"User query:\n{user_query}\n\n"
        f"Current tool stage:\n{tool_stage or 'unknown'}\n\n"
        f"Latest tool execution result JSON:\n{execution_json}\n\n"
        f"Retry reason (if any):\n{retry_reason or 'None'}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
