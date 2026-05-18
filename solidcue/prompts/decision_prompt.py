import json
from typing import Any

from solidcue.prompts.decision_system_prompt import build_decision_system_prompt
from solidcue.agents.configs.loader import load_agent_skill, load_agent_tools
from solidcue.tools.loader import load_tool


def _compact_tool_description(description: str) -> str:
    text = " ".join((description or "").split())
    if not text:
        return "Use this when appropriate."
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
_EXPENSIVE_VALUE_MAX_CHARS = 200
_NORMAL_VALUE_MAX_CHARS = 500


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


def _build_time_location_context(meta: dict[str, Any]) -> dict[str, str]:
    return {
        "current_time": str(meta.get("current_time", "Unknown time")),
        "location": str(meta.get("location", "Unknown location")),
        "timezone": str(meta.get("timezone", "Unknown timezone")),
        "current_time_utc": str(meta.get("current_time_utc", "Unknown UTC time")),
    }


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

    context_lines: list[str] = []
    for key, value in task_context.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            value_text = str(value)
        else:
            value_text = json.dumps(value, ensure_ascii=True, default=str)
        context_lines.append(f"- {key_text}: {value_text}")
    task_context_text = "\n".join(context_lines) if context_lines else "- none"

    phase = str(meta.get("phase") or "source")
    phase_instruction = {
        "source": "You MUST use a tool to gather evidence. Responding with a final answer is only valid if no tool can help.",
        "artifact": "You MUST select the artifact tool to create the output. Do not respond with text.",
        "synthesis": "Compose a final response from the evidence gathered. No tool use needed.",
    }.get(phase, "Use a tool if evidence is needed, otherwise respond.")

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

    skill_section_ref = task_context.get("follow_skill_section") if isinstance(task_context, dict) else None
    skill_delegation = ""
    if isinstance(skill_section_ref, str) and skill_section_ref.strip():
        skill_delegation = (
            f"\n=== SKILL.md DELEGATION ===\n"
            f"This task references SKILL.md section: '{skill_section_ref.strip()}'\n"
            f"Read that section in the Skill guidance below for:\n"
            f"- Field/column specifications\n"
            f"- Default values and value mappings\n"
            f"- Naming patterns and formats\n"
            f"You MUST follow the section's specifications exactly when constructing tool inputs.\n"
            f"If the section is missing or unclear, respond with action=respond explaining the gap.\n"
        )

    history_summary = _format_tool_call_history(tool_call_history)
    return (
        f"=== CURRENT TASK DIRECTIVE ===\n"
        f"You are on task {task_id} of {int(task_count)} ({current_task_type or 'unknown'}).\n"
        f"Goal: {task_desc}\n"
        f"Task execution context (authoritative values for tool arguments):\n{task_context_text}\n"
        f"{tool_enforcement}"
        f"{skill_delegation}"
        f"Phase instruction: {phase_instruction}\n"
        f"Do NOT work on any other task.\n"
        f"Do NOT call a tool that already succeeded in a previous task.\n\n"
        f"--- TOOL HISTORY ---\n"
        f"{history_summary}\n"
        f"If a required ID is missing from history above, call the discovery tool to retrieve it."
    )


def _append_path_hints(system_prompt: str, meta: dict[str, Any]) -> str:
    source_paths = meta.get("source_paths")
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

    output_paths = meta.get("output_paths")
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

    source_filenames = meta.get("source_filenames")
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

    output_filenames = meta.get("output_filenames")
    if isinstance(output_filenames, list):
        sanitized_names = [str(name).strip() for name in output_filenames if str(name).strip()]
        if sanitized_names:
            joined_names = "\n".join(f"- {name}" for name in sanitized_names)
            system_prompt += (
                "\n\n=== OUTPUT FILENAMES ===\n"
                "Output filename hints:\n"
                "STRICT REQUIREMENT: Replace all {{placeholders}}, <tags>, or YYYYMMDD with real-world values.\n"
                "1. Identify the specific entities (People, Organizations, Projects, or Roles) mentioned in the Goal or Context.\n"
                "2. Map these specific names to the placeholders. Never use generic category words (like 'candidate', 'user', or 'file') if a specific name is available.\n"
                "3. Ensure the final filename is a fully resolved string with no brackets or braces remaining.\n"
                f"{joined_names}"
            )
    return system_prompt


def _append_agent_static_guidance(system_prompt: str, agent: Any) -> str:
    agent_key = str(getattr(agent, "agent_key", "") or "").strip()
    if not agent_key:
        return system_prompt

    skill_guidance = load_agent_skill(agent_key)
    if skill_guidance:
        system_prompt += (
            "\n\nSkill guidance:\n"
            "Follow these skill-level rules for output format, naming, and execution behavior.\n"
            f"{skill_guidance}"
        )
    tools_guidance = load_agent_tools(agent_key)
    if tools_guidance:
        system_prompt += (
            "\n\nTools routing guidance:\n"
            "Follow these tool-selection rules when choosing retrieval and artifact actions.\n"
            f"{tools_guidance}"
        )
    return system_prompt


def _append_run_state_overrides(
    system_prompt: str,
    *,
    meta: dict[str, Any],
    current_task_type: str,
    retry_reason: str | None,
) -> str:
    synthesis_draft = meta.get("synthesis_draft")
    if isinstance(synthesis_draft, str) and synthesis_draft.strip() and current_task_type == "artifact_generation":
        system_prompt += (
            "\n\nSynthesis content is READY. Call the artifact tool now. "
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


def _build_runtime_context_message(
    *,
    time_location_context: dict[str, str],
    task_guidance: str,
    meta: dict[str, Any],
    current_task_type: str,
    retry_reason: str | None,
) -> str:
    runtime_context = (
        "=== RUNTIME CONTEXT ===\n"
        f"- Time: {time_location_context['current_time']} ({time_location_context['timezone']})\n"
        f"- Location: {time_location_context['location']}\n"
        f"- UTC: {time_location_context['current_time_utc']}\n\n"
        "=== TASK GUIDANCE ===\n"
        f"{task_guidance}"
    )
    runtime_context = _append_path_hints(runtime_context, meta)
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
):
    tools = list(agent.tools or [])
    meta = metadata if isinstance(metadata, dict) else {}
    current_task = meta.get("current_task") if isinstance(meta.get("current_task"), dict) else {}
    current_task_type = str(current_task.get("type") or "")

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
            tool_lines.append(f"- {tool_key}: Use this when appropriate.")

    tool_descriptions = "\n".join(tool_lines)
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
    system_prompt = _append_agent_static_guidance(system_prompt, agent)
    runtime_context = _build_runtime_context_message(
        time_location_context=time_location_context,
        task_guidance=task_guidance,
        meta=meta,
        current_task_type=current_task_type,
        retry_reason=retry_reason,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": runtime_context},
        {"role": "user", "content": user_input},
    ]

    return messages
