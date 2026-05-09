def build_synthesis_system_prompt(
    *,
    current_time: str,
    timezone: str,
    location: str,
    persona_text: str | None = None,
) -> str:
    prompt = (
        "You are an Editor. Rewrite raw research notes into a polished, user-ready response. "
        "Do not invent facts. Preserve factual content. Improve clarity, structure, and tone. "
        f"Current context is {current_time} ({timezone}) in {location}. "
        "Mention time/date only when relevant to the user query."
    )
    if persona_text:
        prompt += (
            "\n\nPersona guidance:\n"
            "Apply this persona only to the user-facing response. Do not expose internal planning JSON.\n"
            f"{persona_text}"
        )
    return prompt
