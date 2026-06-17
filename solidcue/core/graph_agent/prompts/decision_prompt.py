import json
from typing import Any

from solidcue.core.graph_agent.prompts.decision_system_prompt import build_decision_system_prompt
from solidcue.agent_configs.loader import load_agent_skill, load_agent_tools
from solidcue.tools.loader import load_tool

"""
Decision Prompt Builder
-----------------------

This module builds the messages sent to the decision LLM.
It keeps runtime context compact by:
1) Summarizing tool history with truncation rules
2) Injecting phase-scoped static guidance (SKILL/TOOLS)
3) Rendering only the current-task directive and relevant hints
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOOL_DESCRIPTION = "Use this when appropriate."

PHASE_INSTRUCTION: dict[str, str] = {
    "source": "You MUST use a tool to gather evidence. Responding with a final answer is only valid if no tool can help.",
    "artifact": "You MUST select the artifact tool to create the output. Do not respond with text.",
    "synthesis": "Compose a final response from the evidence gathered. No tool use needed.",
}

PHASE_WITH_SKILL_GUIDANCE = {"artifact", "source", "synthesis"}
PHASE_WITH_TOOLS_GUIDANCE = {"artifact", "source"}

TIME_LOCATION_DEFAULTS = {
    "current_time": "Unknown time",
    "current_date": "Unknown date",
    "location": "Unknown location",
    "timezone": "Unknown timezone",
}


def _compact_tool_description(description: str) -> str:
    text = " ".join((description or "").split())
    if not text:
        return DEFAULT_TOOL_DESCRIPTION
    if "." in text:
        text = text.split(".", 1)[0].strip()
    if len(text) > 140:
        text = text[:139].rstrip() + "…"
    return text


def _format_parameter(name: str, prop: dict[str, Any], is_required: bool) -> str:
    type_str = _resolve_type(prop)
    parts = [f"    {name} ({type_str}"]
    if is_required:
        parts.append(", required")
    default = prop.get("default")
    if default is not None and not is_required:
        default_repr = json.dumps(default) if not isinstance(default, str) else default
        parts.append(f", default: {default_repr}")
    parts.append(")")
    desc = prop.get("description")
    if desc:
        short = " ".join(desc.split())
        if len(short) > 120:
            short = short[:119].rstrip() + "…"
        parts.append(f": {short}")
    return "".join(parts)


def _resolve_type(prop: dict[str, Any]) -> str:
    if "anyOf" in prop:
        types = []
        for variant in prop["anyOf"]:
            if isinstance(variant, dict):
                t = variant.get("type")
                if t:
                    types.append(str(t))
        return " | ".join(types) if types else "any"
    return str(prop.get("type", "any"))


# Keys that typically hold large content - truncate aggressively to save tokens
_EXPENSIVE_KEYS = {
    "text",
    "content",
    "html",
    "body",
    "markdown",
    "content_base64",
    "raw",
    "file_content",
    "data",
}
_EXPENSIVE_VALUE_MAX_CHARS = 2000
_NORMAL_VALUE_MAX_CHARS = 5000
_PHASE_GUIDANCE_MAX_CHARS = 50000


def _truncate_value(key: str, value: Any) -> Any:
    """Truncate or strip large string values to save tokens.

    Expensive keys (content, content_base64, etc.) are truncated to a short
    preview so the LLM retains key identifiers (names, titles, IDs).
    """
    if isinstance(value, str):
        if key.lower() in _EXPENSIVE_KEYS and len(value) > _EXPENSIVE_VALUE_MAX_CHARS:
            return value[:_EXPENSIVE_VALUE_MAX_CHARS].rstrip() + "… [truncated]"
        if len(value) > _NORMAL_VALUE_MAX_CHARS:
            return value[: _NORMAL_VALUE_MAX_CHARS - 1].rstrip() + "…"
        return value
    if isinstance(value, dict):
        return {k: _truncate_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_value(key, v) for v in value]
    return value


def _format_compact_value(value: Any) -> str:
    """Format value as compact JSON with expensive keys truncated."""
    truncated = _truncate_value("", value) if not isinstance(value, dict) else {
        k: _truncate_value(k, v) for k, v in value.items()
    }
    try:
        return json.dumps(truncated, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(truncated)[:300]


def _format_tool_call_history(tool_call_history: list[dict[str, Any]] | None) -> str:
    """Format tool_call_history into a compact summary for the decision prompt.

    Includes tool inputs, execution content, and accomplishments so the LLM can reference actual data
    from previous tasks (e.g., IDs, extracted content) and what was accomplished.
    Truncates expensive content keys (text, content, base64) to save tokens.
    """
    if not isinstance(tool_call_history, list) or not tool_call_history:
        return "None"

    lines: list[str] = []
    for entry in tool_call_history:
        if not isinstance(entry, dict):
            continue
        tool_name = entry.get("tool_name", "unknown")
        tool_input = entry.get("tool_input") or {}
        success = entry.get("success")
        status = "✓" if success else "✗"

        # Truncate tool_input (especially for tools with large content params)
        truncated_input = _format_compact_value(tool_input) if tool_input else "{}"
        line = f"{status} {tool_name}({truncated_input})"

        execution_result = entry.get("execution_result")
        output = execution_result.get("content") if isinstance(execution_result, dict) else None
        if output is not None:
            output_str = _format_compact_value(output).strip()
            if output_str and output_str != "{}":
                line += f" → {output_str}"

        accomplishments = entry.get("accomplishments")
        if isinstance(accomplishments, list) and accomplishments:
            met = [a for a in accomplishments if isinstance(a, str) and a.endswith("_met")]
            missing = [a for a in accomplishments if isinstance(a, str) and a.endswith("_missing")]
            acc_parts = []
            if met:
                acc_parts.append("✓ " + ", ".join(a[:-4] for a in met))  # Remove "_met" suffix
            if missing:
                acc_parts.append("✗ " + ", ".join(a[:-8] for a in missing))  # Remove "_missing" suffix
            if acc_parts:
                line += f" [{', '.join(acc_parts)}]"

        lines.append(line)
    return "\n".join(lines) if lines else "None"


def _format_conversation_history(chat_history: list[dict[str, Any]] | None, *, limit: int = 8) -> str:
    if not isinstance(chat_history, list) or not chat_history:
        return "None"

    lines: list[str] = []
    for entry in chat_history[-limit:]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip()
        content = str(entry.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


def _build_time_location_context(meta: dict[str, Any]) -> dict[str, str]:
    return {
        "current_time": str(meta.get("current_time", TIME_LOCATION_DEFAULTS["current_time"])),
        "current_date": str(meta.get("current_date", TIME_LOCATION_DEFAULTS["current_date"])),
        "location": str(meta.get("location", TIME_LOCATION_DEFAULTS["location"])),
        "timezone": str(meta.get("timezone", TIME_LOCATION_DEFAULTS["timezone"])),
    }


def _stringify_context_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return json.dumps(value, ensure_ascii=True, default=str)


def _format_task_context(task_context: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in task_context.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        lines.append(f"- {key_text}: {_stringify_context_value(value)}")
    return "\n".join(lines) if lines else "- none"


def _phase_instruction_for(meta: dict[str, Any]) -> str:
    phase = str(meta.get("phase") or "source")
    return PHASE_INSTRUCTION.get(phase, "Use a tool if evidence is needed, otherwise respond.")


def _build_task_guidance(
    *,
    meta: dict[str, Any],
    current_task: dict[str, Any],
    current_task_type: str,
    tool_call_history: list[dict[str, Any]] | None,
) -> str:
    task_id = meta.get("current_task_id", "task_1")
    task_count = meta.get("total_tasks", 1)
    task_desc = str(current_task.get("description", "No active task"))
    task_context = current_task.get("context") if isinstance(current_task.get("context"), dict) else {}
    task_context_text = _format_task_context(task_context)
    phase_instruction = _phase_instruction_for(meta)

    planned_tool = task_context.get("tool") if isinstance(task_context, dict) else None
    tool_enforcement = ""
    if isinstance(planned_tool, str) and planned_tool.strip():
        tool_enforcement = (
            f"\n=== MANDATORY TOOL SELECTION ===\n"
            f"You MUST use this exact tool: {planned_tool.strip()}\n"
            f"Do NOT substitute with another tool. The task plan is the contract.\n"
            f"If you cannot use this tool (e.g., missing inputs), respond with action=respond\n"
            f"explaining the blocker, and the router will handle the retry.\n"
        )

    history_summary = _format_tool_call_history(tool_call_history)
    return (
        f"=== CURRENT TASK DIRECTIVE ===\n"
        f"You are on task {task_id} of {int(task_count)} ({current_task_type or 'unknown'}).\n"
        f"Goal: {task_desc}\n"
        f"Task execution context (authoritative values for tool arguments):\n{task_context_text}\n"
        f"{tool_enforcement}"
        f"Phase instruction: {phase_instruction}\n"
        f"Do NOT work on any other task.\n"
        f"Do NOT call a tool that already succeeded in a previous task.\n\n"
        f"--- TOOL HISTORY ---\n"
        f"{history_summary}\n"
        f"If a required ID is missing from history above, call the discovery tool to retrieve it."
    )


def _append_path_hints(
    system_prompt: str,
    source_paths: list[str] | None = None,
    output_paths: list[str] | None = None,
    source_filenames: list[str] | None = None,
    output_filenames: list[str] | None = None,
) -> str:
    if isinstance(source_paths, list):
        sanitized_paths = [str(path).strip() for path in source_paths if str(path).strip()]
        if sanitized_paths:
            joined_paths = "\n".join(f"- {path}" for path in sanitized_paths)
            system_prompt += (
                "\n\n=== SOURCE PATHS ===\n"
                "Source path hints:\n"
                "You MUST treat these paths as the primary locations for file discovery. "
                "Replace any placeholders like {{folder_name}} or <path> with actual values from the task context.\n"
                f"{joined_paths}"
            )

    if isinstance(output_paths, list):
        sanitized_paths = [str(path).strip() for path in output_paths if str(path).strip()]
        if sanitized_paths:
            joined_paths = "\n".join(f"- {path}" for path in sanitized_paths)
            system_prompt += (
                "\n\n=== OUTPUT PATHS ===\n"
                "Output path hints:\n"
                "Follow this convention STRICTLY as the destination for generated artifacts. "
                "Resolve any {{variables}} or <placeholders> using task context before execution.\n"
                f"{joined_paths}"
            )

    if isinstance(source_filenames, list):
        sanitized_names = [str(name).strip() for name in source_filenames if str(name).strip()]
        if sanitized_names:
            joined_names = "\n".join(f"- {name}" for name in sanitized_names)
            system_prompt += (
                "\n\n=== SOURCE FILENAMES ===\n"
                "Source filename hints:\n"
                "Use these exact filename patterns when locating input files. "
                "Replace {{tags}} or <tags> with real data (e.g., replace {{candidate_name}} with the actual name).\n"
                f"{joined_names}"
            )

    if isinstance(output_filenames, list):
        sanitized_names = [str(name).strip() for name in output_filenames if str(name).strip()]
        if sanitized_names:
            joined_names = "\n".join(f"- {name}" for name in sanitized_names)
            system_prompt += (
                "\n\n=== OUTPUT FILENAMES ===\n"
                "Output filename hints:\n"
                "STRICT REQUIREMENT: Replace all {{placeholders}}, <tags>, or YYYYMMDD with real-world values.\n"
                "1. Identify the specific entities (People, Organizations, Projects, or Roles). Check the user's original request FIRST, then the Goal or Context. The user's request is the primary source for target entities (e.g., company, role, recipient).\n"
                "2. Map these specific names to the placeholders. Never use generic category words (like 'candidate', 'user', or 'file') if a specific name is available.\n"
                "3. Ensure the final filename is a fully resolved string with no brackets or braces remaining.\n"
                f"{joined_names}"
            )
    return system_prompt


def _truncate_guidance(text: str, max_chars: int = _PHASE_GUIDANCE_MAX_CHARS) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "… [truncated]"


def _append_agent_static_guidance(system_prompt: str, agent: Any, phase: str) -> str:
    agent_key = str(getattr(agent, "agent_key", "") or "").strip()
    if not agent_key:
        return system_prompt

    normalized_phase = (phase or "source").strip().casefold()
    include_skill = normalized_phase in PHASE_WITH_SKILL_GUIDANCE
    include_tools = normalized_phase in PHASE_WITH_TOOLS_GUIDANCE

    skill_guidance = load_agent_skill(agent_key)
    if include_skill and skill_guidance:
        skill_text = _truncate_guidance(skill_guidance)
        system_prompt += (
            f"\n\nSkill guidance (phase={normalized_phase}):\n"
            "Follow these skill-level rules for output format, naming, and execution behavior.\n"
            f"{skill_text}"
        )

    tools_guidance = load_agent_tools(agent_key)
    if include_tools and tools_guidance:
        tools_text = _truncate_guidance(tools_guidance)
        system_prompt += (
            f"\n\nTools routing guidance (phase={normalized_phase}):\n"
            "Follow these tool-selection rules when choosing retrieval and artifact actions.\n"
            f"{tools_text}"
        )
    return system_prompt


def _build_tool_descriptions(tools: list[str]) -> str:
    tool_lines: list[str] = []
    for tool_key in tools:
        try:
            tool_config = load_tool(tool_key)
            description = _compact_tool_description(tool_config.description.strip())
            schema = getattr(getattr(tool_config, "mcp", None), "input_schema", None)
            properties = schema.get("properties") if isinstance(schema, dict) else None
            required_set = set(schema.get("required", [])) if isinstance(schema, dict) else set()

            if isinstance(properties, dict) and properties:
                param_lines = [
                    _format_parameter(name, prop, name in required_set)
                    for name, prop in properties.items()
                    if isinstance(prop, dict)
                ]
                tool_lines.append(
                    f"- {tool_key}: {description}\n"
                    f"  Parameters:\n" + "\n".join(param_lines)
                )
            else:
                tool_lines.append(f"- {tool_key}: {description}")
        except Exception:
            tool_lines.append(f"- {tool_key}: {DEFAULT_TOOL_DESCRIPTION}")

    return "\n".join(tool_lines)


def _append_run_state_overrides(
    system_prompt: str,
    *,
    meta: dict[str, Any],
    current_task_type: str,
    retry_reason: str | None,
) -> str:
    if current_task_type == "artifact_generation":
        system_prompt += (
            "\n\nArtifact content is READY via handoff. Call the artifact tool now. "
            "For the content parameter, use the exact string: [payload via handoff] "
            "— the execution layer will replace it with the real content. "
            "Do NOT fetch or list source files again."
        )

    if retry_reason:
        system_prompt += (
            "\n\n=== TASK STATUS: INCOMPLETE ===\n"
            f"{retry_reason}\n"
        )
    return system_prompt


def _format_target_artifacts_source(items: list[dict[str, Any]] | None) -> str:
    if not isinstance(items, list) or not items:
        return ""
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("index", "?")
        source_type = item.get("source_type", "")
        source_ref = item.get("source_ref", "")
        item_key = item.get("item_key", "")
        if source_ref and item_key:
            lines.append(f"- [{idx}] type={source_type} ref={source_ref} item_key={item_key}")
    return "\n".join(lines)


def _build_runtime_context_message(
    *,
    time_location_context: dict[str, str],
    task_guidance: str,
    meta: dict[str, Any],
    current_task_type: str,
    retry_reason: str | None,
    chat_history: list[dict[str, Any]] | None,
    source_paths: list[str] | None = None,
    output_paths: list[str] | None = None,
    source_filenames: list[str] | None = None,
    output_filenames: list[str] | None = None,
    target_artifacts_source: list[dict[str, Any]] | None = None,
) -> str:
    runtime_context = (
        "=== RUNTIME CONTEXT ===\n"
        f"- Date: {time_location_context['current_date']}\n"
        f"- Time: {time_location_context['current_time']} ({time_location_context['timezone']})\n"
        f"- Location: {time_location_context['location']}\n\n"
        "=== RECENT CONVERSATION ===\n"
        f"{_format_conversation_history(chat_history)}\n\n"
        "=== TASK GUIDANCE ===\n"
        f"{task_guidance}"
    )
    artifacts_text = _format_target_artifacts_source(target_artifacts_source)
    if artifacts_text:
        runtime_context += (
            "\n\n=== SOURCE INPUTS ===\n"
            "These are the user-provided source inputs for this task. "
            "Use source_ref as the URL or path argument when calling tools. "
            "Match tasks to sources using item_key.\n"
            f"{artifacts_text}"
        )
    runtime_context = _append_path_hints(
        runtime_context,
        source_paths=source_paths,
        output_paths=output_paths,
        source_filenames=source_filenames,
        output_filenames=output_filenames,
    )
    runtime_context = _append_run_state_overrides(
        runtime_context,
        meta=meta,
        current_task_type=current_task_type,
        retry_reason=retry_reason,
    )
    return runtime_context


def build_decision_messages(
    agent,
    user_input: str,
    retry_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    tool_call_history: list[dict[str, Any]] | None = None,
    chat_history: list[dict[str, Any]] | None = None,
    source_paths: list[str] | None = None,
    output_paths: list[str] | None = None,
    source_filenames: list[str] | None = None,
    output_filenames: list[str] | None = None,
    target_artifacts_source: list[dict[str, Any]] | None = None,
):
    tools = list(agent.tools or [])
    meta = metadata if isinstance(metadata, dict) else {}
    phase = str(meta.get("phase") or "source")
    current_task = meta.get("current_task") if isinstance(meta.get("current_task"), dict) else {}
    current_task_type = str(current_task.get("type") or "")
    tool_descriptions = _build_tool_descriptions(tools)
    time_location_context = _build_time_location_context(meta)
    task_guidance = _build_task_guidance(
        meta=meta,
        current_task=current_task,
        current_task_type=current_task_type,
        tool_call_history=tool_call_history,
    )

    system_prompt = build_decision_system_prompt(
        agent_name=agent.name,
        agent_description=agent.description,
        tool_descriptions=tool_descriptions,
    )
    system_prompt = _append_agent_static_guidance(system_prompt, agent, phase)
    runtime_context = _build_runtime_context_message(
        time_location_context=time_location_context,
        task_guidance=task_guidance,
        meta=meta,
        current_task_type=current_task_type,
        retry_reason=retry_reason,
        chat_history=chat_history,
        source_paths=source_paths,
        output_paths=output_paths,
        source_filenames=source_filenames,
        output_filenames=output_filenames,
        target_artifacts_source=target_artifacts_source,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": runtime_context},
        {"role": "user", "content": user_input},
    ]

    return messages
