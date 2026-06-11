from __future__ import annotations

from typing import Any

from solidcue.core.graph_agent.prompts.planning_system_prompt import build_planning_system_prompt


def _compact_guidance(text: str, limit: int = 15000) -> str:
    """Compact guidance text by stripping empty lines.

    Default limit 15000 chars accommodates full SKILL.md files without
    truncating critical sections (filename formats, output specifications,
    field definitions) that typically appear at the end of these files.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    compact = "\n".join(line.rstrip() for line in raw.splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _format_conversation_history(chat_history: list[dict[str, Any]] | None, *, limit: int = 8) -> str:
    if not isinstance(chat_history, list) or not chat_history:
        return "None specified."

    lines: list[str] = []
    for entry in chat_history[-limit:]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip()
        content = str(entry.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        lines.append(f"- {role}: {content}")
    return "\n".join(lines) if lines else "None specified."


def build_planning_messages(
    *,
    user_input: str,
    skill_guidance: str = "",
    tools_guidance: str = "",
    metadata: dict[str, Any] | None = None,
    chat_history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    # 1. Prepare dynamic components
    compact_skill = _compact_guidance(skill_guidance) or "General technical project management standards."
    compact_tools = _compact_guidance(tools_guidance) or "No specific tool constraints."
    
    source_paths_str = "None specified."
    output_paths_str = "None specified."
    source_filenames_str = "None specified."
    output_filenames_str = "None specified."
    if isinstance(metadata, dict):
        source_paths = metadata.get("source_paths")
        if isinstance(source_paths, list):
            paths = [str(p).strip() for p in source_paths if str(p).strip()]
            if paths:
                source_paths_str = "\n".join(f"- {p}" for p in paths)
        output_paths = metadata.get("output_paths")
        if isinstance(output_paths, list):
            paths = [str(p).strip() for p in output_paths if str(p).strip()]
            if paths:
                output_paths_str = "\n".join(f"- {p}" for p in paths)
        source_filenames = metadata.get("source_filenames")
        if isinstance(source_filenames, list):
            filenames = [str(name).strip() for name in source_filenames if str(name).strip()]
            if filenames:
                source_filenames_str = "\n".join(f"- {name}" for name in filenames)
        output_filenames = metadata.get("output_filenames")
        if isinstance(output_filenames, list):
            filenames = [str(name).strip() for name in output_filenames if str(name).strip()]
            if filenames:
                output_filenames_str = "\n".join(f"- {name}" for name in filenames)

    # 2. Build static system prompt + dynamic runtime context
    system_prompt = build_planning_system_prompt()
    runtime_context = f"""
=== RUNTIME CONTEXT ===
**CRITICAL: The Skill Guidance below is the authoritative source for output formats, naming patterns, field structures, and domain-specific rules. Follow it exactly. Do NOT invent or simplify specifications.**

- **Skill Guidance:**
{compact_skill}

- **Tooling Constraints:** {compact_tools}
- **Preferred Source Paths:**
{source_paths_str}
- **Preferred Output Paths:**
{output_paths_str}
- **Preferred Source Filenames:**
{source_filenames_str}
- **Preferred Output Filenames:**
{output_filenames_str}
- **Recent Conversation:**
{_format_conversation_history(chat_history)}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": runtime_context},
        {"role": "user", "content": f"User request: {user_input}"},
    ]
