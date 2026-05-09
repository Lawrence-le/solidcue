from __future__ import annotations

import json
import uuid
from typing import Any, cast

from solidcue.agents.configs.loader import load_agent, load_agent_persona
from solidcue.core.state.schema import AgentState
from solidcue.prompts.artifact_generation_prompt import build_artifact_generation_messages
from solidcue.providers.factory import get_provider
from solidcue.tools.loader import load_tool
from solidcue.tools.schema import ToolConfig
from solidcue.tools.stages import GENERATABLE_TOOL_FIELDS, get_required_tool_fields

ARTIFACT_GENERATION_FALLBACK = (
    "I couldn't generate the required artifact content safely. Please retry with more specific details."
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        marker_end = stripped.find("\n")
        fence_end = stripped.rfind("```")
        if marker_end != -1 and fence_end > marker_end:
            stripped = stripped[marker_end:fence_end].strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _artifact_generation_fields(tool: ToolConfig) -> list[str]:
    required_fields = get_required_tool_fields(tool)
    fields = list(required_fields)

    schema = getattr(getattr(tool, "mcp", None), "input_schema", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    property_map = properties if isinstance(properties, dict) else {}

    for field in sorted(GENERATABLE_TOOL_FIELDS):
        if field in property_map and field not in fields:
            fields.append(field)

    return fields


def _missing_generation_fields(tool: ToolConfig, tool_input: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in _artifact_generation_fields(tool):
        value = tool_input.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
            continue
        if field == "values" and isinstance(value, list) and not value:
            missing.append(field)
    return missing


def artifact_generation_node(state: AgentState) -> dict[str, Any]:
    decision = cast(dict[str, Any], state.get("decision") or {})
    artifact_plan = cast(dict[str, Any], state.get("artifact_plan") or {})
    if not decision and artifact_plan:
        decision = {
            "action": "use_tool",
            "tool_stage": "artifact",
            "tool_name": artifact_plan.get("tool_name"),
            "tool_input": artifact_plan.get("tool_input") or {},
            "final_answer": None,
            "approval_preview": None,
        }
    if decision.get("action") != "use_tool" or decision.get("tool_stage") != "artifact":
        return {}

    tool_name = decision.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return {}

    existing_tool_input = decision.get("tool_input")
    tool_input = existing_tool_input if isinstance(existing_tool_input, dict) else {}
    context_evidence_parts: list[str] = []
    stored_context_evidence = state.get("context_evidence")
    if isinstance(stored_context_evidence, list):
        for item in stored_context_evidence:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if content is None:
                continue
            text = str(content).strip()
            if text:
                context_evidence_parts.append(text)
    context_evidence = "\n\n".join(part for part in context_evidence_parts if part.strip())

    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {}

    try:
        agent = load_agent(agent_key)
        tool = load_tool(tool_name)
        messages = build_artifact_generation_messages(
            user_query=str(state.get("user_input", "")),
            tool_name=tool_name,
            tool_description=tool.description,
            existing_tool_input=tool_input,
            required_fields=_artifact_generation_fields(tool),
            context_evidence=context_evidence,
            metadata=state.get("metadata"),
            persona_text=load_agent_persona(agent_key),
        )
        output = get_provider(agent.provider).generate(messages)
        generated_input = _extract_json_object(str(output or ""))
        if not generated_input:
            return _artifact_generation_failure(messages)

        merged_input = {**tool_input, **generated_input}
        missing_fields = _missing_generation_fields(tool, merged_input)
        if missing_fields:
            return _artifact_generation_failure(messages)

        updated_decision = {**decision, "tool_input": merged_input}
        return {
            "decision": updated_decision,
            "artifact_input": merged_input,
            "artifact_generation_messages": messages,
            "latest_output": str(output or ""),
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(merged_input),
                            },
                        }
                    ],
                }
            ],
        }
    except Exception:
        return _artifact_generation_failure()


def _artifact_generation_failure(messages: list[dict[str, str]] | None = None) -> dict[str, Any]:
    update: dict[str, Any] = {
        "tool_use": False,
        "decision": {
            "action": "respond",
            "thought": "Artifact generation failed before tool execution.",
            "tool_stage": None,
            "tool_name": None,
            "tool_input": {},
            "final_answer": ARTIFACT_GENERATION_FALLBACK,
            "approval_preview": None,
        },
        "draft_output": ARTIFACT_GENERATION_FALLBACK,
        "finalization_reason": "artifact_generation_failed",
    }
    if messages is not None:
        update["artifact_generation_messages"] = messages
    return update
