
from typing import Any

from solidcue.prompts.decision_system_prompt import build_decision_system_prompt
from solidcue.tools.stages import infer_tool_stage
from solidcue.tools.loader import load_mcp_server, load_tool


def build_decision_messages(
    agent,
    user_input: str,
    retry_reason: str | None = None,
    transcript: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
):
    tools = agent.tools or []

    tool_lines: list[str] = []
    for tool_key in tools:
        try:
            tool_config = load_tool(tool_key)
            description = tool_config.description.strip() or "Use this when appropriate."
            tool_stage = infer_tool_stage(tool_key, tool_config)
            mcp_source_line = None

            if tool_config.type == "mcp" and tool_config.mcp:
                server_key = tool_config.mcp.server_key
                source_line = f"mcp::{server_key}"
                try:
                    server = load_mcp_server(server_key)
                    source_line = f"{source_line} ({server.name})"
                except Exception:
                    pass
                mcp_source_line = f"  - Source: {source_line}"

            if tool_config.type == "mcp" and tool_config.mcp and tool_config.mcp.input_schema:
                schema = tool_config.mcp.input_schema
                required_fields = schema.get("required", [])
                required_text = (
                    ", ".join(required_fields)
                    if isinstance(required_fields, list) and required_fields
                    else "none"
                )
                tool_lines.append(
                    f"- {tool_key}: {description}\n"
                    f"  - Tool stage: {tool_stage}\n"
                    f"{mcp_source_line + chr(10) if mcp_source_line else ''}"
                    f"  - Required tool_input fields: {required_text}"
                )
            else:
                tool_line = f"- {tool_key}: {description}"
                if mcp_source_line:
                    tool_line = f"{tool_line}\n{mcp_source_line}"
                tool_line = (
                    f"{tool_line}\n"
                    f"  - Tool stage: {tool_stage}"
                )
                tool_lines.append(tool_line)
        except Exception:
            tool_lines.append(f"- {tool_key}: Use this when appropriate.")

    tool_descriptions = "\n".join(tool_lines)
    meta = metadata if isinstance(metadata, dict) else {}
    current_time = meta.get("current_time", "Unknown time")
    location = meta.get("location", "Unknown location")
    timezone = meta.get("timezone", "Unknown timezone")
    current_time_utc = meta.get("current_time_utc", "Unknown UTC time")

    system_prompt = build_decision_system_prompt(
        agent_name=agent.name,
        agent_description=agent.description,
        current_time=current_time,
        timezone=timezone,
        location=location,
        tool_descriptions=tool_descriptions,
        current_time_utc=current_time_utc,
    )

    persona_source_paths = meta.get("persona_source_paths")
    if isinstance(persona_source_paths, list):
        sanitized_paths = [str(path).strip() for path in persona_source_paths if str(path).strip()]
        if sanitized_paths:
            joined_paths = "\n".join(f"- {path}" for path in sanitized_paths)
            system_prompt += (
                "\n\nPersona source path hints:\n"
                "You MUST treat these Google Drive paths as the first source locations for file discovery.\n"
                "Before broader Drive search, try these paths directly when relevant to the request.\n"
                f"{joined_paths}"
            )

    if retry_reason:
        system_prompt += (
            "\n\nPrevious attempt failed.\n"
            f"Failure reason: {retry_reason}\n"
            "If another available tool can satisfy the same user request, use it instead. "
            "Do not repeat a failed tool call unless you have a specific different input that is likely to fix the failure. "
            "If previous successful tool outputs did not contain the requested fact and no untried suitable tool remains, respond with the specific limitation. "
            "Respond with a limitation only when no available tool can help. "
            "Keep final_answer concise and user-facing; do not mention internal tool names, HTTP status codes, provider names, stack traces, or raw errors. "
            "Return corrected JSON only and keep the same contract keys."
        )

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if transcript:
        messages.extend(transcript)
    else:
        messages.append({"role": "user", "content": user_input})

    return messages
