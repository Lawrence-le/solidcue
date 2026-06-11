from typing import Any

from solidcue.core.graph_agent.prompts.synthesis_system_prompt import build_synthesis_system_prompt


def build_synthesis_messages(
    *,
    user_query: str,
    raw_data: str,
    metadata: dict[str, Any] | None = None,
    retry_reason: str | None = None,
    persona_text: str | None = None,
    skill_text: str | None = None,
    task_description: str | None = None,
) -> list[dict[str, str]]:
    meta = metadata if isinstance(metadata, dict) else {}
    current_time = meta.get("current_time", "Unknown time")
    location = meta.get("location", "Unknown location")
    timezone = meta.get("timezone", "Unknown timezone")

    system_prompt = build_synthesis_system_prompt(
        persona_text=persona_text,
        skill_text=skill_text,
    )

    runtime_context = (
        "=== RUNTIME CONTEXT ===\n"
        f"- Time: {current_time} ({timezone})\n"
        f"- Location: {location}\n"
    )
    if task_description:
        runtime_context += (
            "\n# CURRENT TASK\n"
            f"{task_description}\n\n"
            "# OUTPUT BOUNDARY (HARD RULE)\n"
            "Your output must contain ONLY the content described in CURRENT TASK and structured by SKILL REFERENCE. "
            "Do NOT extract, summarize, or append any data for other workflow steps (e.g. spreadsheet rows, tracker fields, metadata tables). "
            "Other tasks in the pipeline will handle their own data extraction — your job is solely to produce the content above. "
            "Any content outside the skill reference structure is a violation.\n"
        )

    synthesis_input = (
        f"USER_QUERY: {user_query}\n\n"
        f"EVIDENCE:\n{raw_data}"
    )

    if retry_reason:
        synthesis_input += (
            f"\n\nPREVIOUS_VALIDATION_FAILURE: {retry_reason}\n"
            "Remove or revise any factual claim that is not directly supported by the provided evidence.\n"
            "Do not add new factual claims unless they are explicitly present in evidence."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": runtime_context},
        {"role": "user", "content": synthesis_input},
    ]
