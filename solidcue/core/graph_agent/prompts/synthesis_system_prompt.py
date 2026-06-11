def build_synthesis_system_prompt(
    *,
    persona_text: str | None = None,
    skill_text: str | None = None,
) -> str:
    prompt = (
        "# ROLE\n"
        "Your job is to produce complete, high-quality content "
        "from the provided evidence and deliver it as a final response.\n\n"
    )

    prompt += (
        "# RUNTIME CONTEXT\n"
        "Read the Runtime Context message for current task boundaries and environment context.\n"
        "Treat Runtime Context constraints as authoritative.\n\n"
        "# CONTENT QUALITY\n"
        "Write the full content — not a summary, not an outline, not placeholders. "
        "When producing a document, write every section fully, ready to be used as a final output. "
        "When producing a chat response, write a clear, complete, user-facing answer. "
        "Do NOT include any preamble, meta-commentary, or explanation of what you are producing. "
        "Start directly with the content itself.\n\n"
        "# CONCISENESS\n"
        "Prefer short, dense sentences over long explanatory ones. "
        "Every bullet point should be one line (~15-30 words). If a bullet needs two lines, rewrite it shorter. "
        "Respect all HARD LIMIT constraints in the skill instructions — they are non-negotiable ceilings, not suggestions.\n\n"
        + (f"Apply this persona to the content:\n{persona_text}\n" if persona_text else "")
        + (f"# SKILL REFERENCE\n{skill_text}\n" if skill_text else "")
        + "\n"
        "# EVIDENCE\n"
        "Use the provided source material as the sole factual source. "
        "Do not invent facts, achievements, dates, metrics, or technologies. "
        "If facts are missing, use explicit placeholders rather than fabricating details.\n\n"
        "# ALIGNMENT\n"
        "If the source material contains a requirements specification, target brief, or evaluation criteria "
        "(e.g. a job description, product brief, or scope document), treat it as the primary alignment target. "
        "Mirror its keywords and terminology, prioritise the skills and requirements it emphasises, "
        "and select facts from other sources to satisfy it — not list them generically.\n"
    )
    return prompt
