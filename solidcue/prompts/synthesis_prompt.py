from typing import Any

from solidcue.prompts.synthesis_system_prompt import build_synthesis_system_prompt


def build_synthesis_messages(
    *,
    user_query: str,
    raw_data: str,
    metadata: dict[str, Any] | None = None,
    retry_reason: str | None = None,
    persona_text: str | None = None,
) -> list[dict[str, str]]:
    meta = metadata if isinstance(metadata, dict) else {}
    current_time = meta.get("current_time", "Unknown time")
    location = meta.get("location", "Unknown location")
    timezone = meta.get("timezone", "Unknown timezone")

    system_prompt = build_synthesis_system_prompt(
        current_time=current_time,
        timezone=timezone,
        location=location,
        persona_text=persona_text,
    )

    user_prompt = (
        f"USER_QUERY: {user_query}\n\n"
        f"RAW_DATA:\n{raw_data}\n\n"
    )

    if retry_reason:
        user_prompt += f"PREVIOUS_VALIDATION_FAILURE: {retry_reason}\nFix that issue in this rewrite.\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
