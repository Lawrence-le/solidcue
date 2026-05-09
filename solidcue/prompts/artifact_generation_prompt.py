from __future__ import annotations

import json
from typing import Any


def build_artifact_generation_messages(
    *,
    user_query: str,
    tool_name: str,
    tool_description: str,
    existing_tool_input: dict[str, Any],
    required_fields: list[str],
    context_evidence: str | None = None,
    metadata: dict[str, Any] | None = None,
    persona_text: str | None = None,
) -> list[dict[str, str]]:
    meta = metadata if isinstance(metadata, dict) else {}
    current_time = meta.get("current_time", "Unknown time")
    location = meta.get("location", "Unknown location")
    timezone = meta.get("timezone", "Unknown timezone")

    system_prompt = (
        "You generate artifact tool arguments. Return exactly one JSON object and no extra text. "
        "Preserve existing tool_input values unless the user request clearly requires improving an empty or placeholder value. "
        "Fill every field listed in REQUIRED_FIELDS, including generated fields such as content or values. "
        "For document artifacts, content must contain the complete document body, not a summary or placeholder. "
        "Use CONTEXT_EVIDENCE as the factual source of truth. Do not invent external facts, achievements, dates, metrics, or technologies. "
        "If source facts are missing, use explicit placeholders instead of fabricating details. "
        f"Current context is {current_time} ({timezone}) in {location}."
    )
    if persona_text:
        system_prompt += (
            "\n\nPersona guidance:\n"
            "Apply this persona to generated artifact content only. Do not expose internal planning JSON.\n"
            f"{persona_text}"
        )

    user_prompt = (
        f"USER_QUERY: {user_query}\n\n"
        f"TOOL_NAME: {tool_name}\n"
        f"TOOL_DESCRIPTION: {tool_description}\n"
        f"REQUIRED_FIELDS: {json.dumps(required_fields, ensure_ascii=True)}\n"
        f"EXISTING_TOOL_INPUT: {json.dumps(existing_tool_input, ensure_ascii=True, default=str)}\n\n"
        f"CONTEXT_EVIDENCE: {context_evidence or ''}\n\n"
        "Return the complete tool_input JSON object for this tool."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
