import json
import re
from typing import TYPE_CHECKING
from typing import Any

from solidcue.agent_configs.loader import load_agent
from solidcue.providers.provider_resolver import get_provider_for_role
from solidcue.core.graph_agent.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_async_stream_generate
from solidcue.core.graph_agent.prompts.final_output_prompt import build_final_output_messages

if TYPE_CHECKING:
    from solidcue.providers.base import BaseProvider

"""
Final Output Node - Function Overview
-------------------------------------

_sanitize_error_text:
Redact sensitive query-parameter values from error/content text.

_parse_json_content:
Parse JSON-like content strings into structured objects when possible.

_summarize_tool_content:
Build short human-readable summary of last tool result payload.

_truncate_content_preview:
Produce bounded preview text for large payloads.

_build_fallback_output:
Build deterministic fallback final response from execution/validation state.

_compact_successful_tool_history:
Compact recent successful tool calls into concise summary lines.

_task_item_key_map / _uploaded_artifacts_by_item:
Build deterministic per-item artifact completion summary from upload tool results.

_llm_compose_user_facing_output:
Generate polished final response via model using current state context.

final_output_node:
Main entrypoint that chooses conversational fast-path, LLM output, or fallback.
"""


_SENSITIVE_QUERY_PARAM_RE = re.compile(r"([?&](?:api_key|key|token|access_token)=)[^&\s]+", re.IGNORECASE)
_MAX_CONTENT_PREVIEW_CHARS = 1200


def _sanitize_error_text(text: str) -> str:
    return _SENSITIVE_QUERY_PARAM_RE.sub(r"\1[redacted]", text)


def _parse_json_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content

    stripped = content.strip()
    if not stripped or stripped[0] not in "{[":
        return content

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return content


def _summarize_tool_content(content: Any) -> str | None:
    parsed = _parse_json_content(content)

    if isinstance(parsed, dict):
        title = parsed.get("title")
        if isinstance(title, str) and title:
            return f"Last tool output title: {title}"

        status = parsed.get("status")
        if status is not None:
            return f"Last tool output status: {status}"

    if isinstance(parsed, str) and parsed.strip():
        return _sanitize_error_text(parsed.strip())

    return None


def _truncate_content_preview(content: Any, max_chars: int = _MAX_CONTENT_PREVIEW_CHARS) -> str:
    if isinstance(content, str):
        text = content
    else:
        try:
            text = json.dumps(content, ensure_ascii=True, default=str)
        except Exception:
            text = str(content)

    text = _sanitize_error_text(text.strip())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


def _build_fallback_output(state: AgentState) -> str:
    retry_reason = state.get("retry_reason")
    execution_result = state.get("execution_result")

    if isinstance(execution_result, dict):
        if execution_result.get("success") is False:
            return (
                "I couldn't retrieve enough reliable information to answer that right now. "
                "Try again later or provide a source to check."
            )

        content_summary = _summarize_tool_content(execution_result.get("content"))
        if content_summary:
            return (
                "I couldn't find the requested information in the available results. "
                "Try again later or provide a source to check."
            )

    if retry_reason:
        return (
            "I couldn't retrieve enough reliable information to answer that right now. "
            "Try again later or provide a source to check."
        )

    return "I couldn't generate a final response for this request."


def _compact_successful_tool_history(state: AgentState) -> list[dict[str, Any]]:
    history = state.get("tool_call_history")
    if not isinstance(history, list) or not history:
        return []

    successful_entries: list[dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        execution_result = entry.get("execution_result")
        if not isinstance(execution_result, dict) or execution_result.get("success") is not True:
            continue

        successful_entries.append(
            {
                "task_id": entry.get("task_id"),
                "tool_name": entry.get("tool_name"),
                "tool_input": entry.get("tool_input") if isinstance(entry.get("tool_input"), dict) else {},
                "accomplishments": entry.get("accomplishments") if isinstance(entry.get("accomplishments"), list) else [],
                "content": _truncate_content_preview(execution_result.get("content")),
            }
        )

    return successful_entries


def _task_item_key_map(state: AgentState) -> dict[str, str]:
    task_plan = state.get("task_plan")
    if not isinstance(task_plan, list):
        return {}

    mapping: dict[str, str] = {}
    for task in task_plan:
        if not isinstance(task, dict):
            continue
        task_id = task.get("id")
        context = task.get("context")
        if not isinstance(task_id, str) or not isinstance(context, dict):
            continue
        item_key = context.get("item_key")
        if isinstance(item_key, str) and item_key.strip():
            mapping[task_id] = item_key.strip()
    return mapping


def _uploaded_artifacts_by_item(state: AgentState) -> list[dict[str, Any]]:
    history = state.get("tool_call_history")
    if not isinstance(history, list) or not history:
        return []

    item_by_task = _task_item_key_map(state)
    uploads: dict[str, dict[str, Any]] = {}
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("tool_name") != "drive_upload_file":
            continue
        execution_result = entry.get("execution_result")
        if not isinstance(execution_result, dict) or execution_result.get("success") is not True:
            continue

        task_id = entry.get("task_id")
        item_key = item_by_task.get(task_id) if isinstance(task_id, str) else ""
        if not item_key:
            continue

        content = execution_result.get("content")
        if not isinstance(content, dict):
            continue
        uploads[item_key] = {
            "item_key": item_key,
            "file_id": content.get("id"),
            "name": content.get("name"),
            "webViewLink": content.get("webViewLink"),
        }

    return list(uploads.values())


def _build_final_output_payload(state: AgentState) -> dict[str, Any] | None:
    successful_tool_history = _compact_successful_tool_history(state)
    if not successful_tool_history:
        return None

    target_artifacts_source = state.get("target_artifacts_source") or []

    return {
        "user_input": str(state.get("user_input") or ""),
        "successful_tool_calls": successful_tool_history,
        "target_artifacts_source": target_artifacts_source,
        "uploaded_artifacts_by_item": _uploaded_artifacts_by_item(state),
    }


def prepare_final_output_stream(state: AgentState) -> tuple["BaseProvider", list[dict[str, str]]] | None:
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return None

    payload = _build_final_output_payload(state)
    if not isinstance(payload, dict):
        return None

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "lite")
        messages = build_final_output_messages(payload)
        return provider, messages
    except Exception:
        return None


def resolve_final_output(state: AgentState, llm_output: str | None = None) -> str:
    existing = state.get("final_response")
    if isinstance(existing, str) and existing.strip():
        existing_output = existing
    else:
        existing_output = None
    return llm_output or state.get("synthesis_draft") or existing_output or _build_fallback_output(state)


async def _llm_compose_user_facing_output(state: AgentState) -> tuple[str | None, dict[str, Any]]:
    prepared = prepare_final_output_stream(state)
    if not prepared:
        return None, {}

    try:
        provider, messages = prepared
        output, metric_final_output = await timed_async_stream_generate(provider, messages, node_name="final_output")
        if isinstance(output, str) and output.strip():
            return output.strip(), metric_final_output
    except Exception:
        return None, {}
    return None, {}


async def final_output_node(state: AgentState) -> dict[str, Any]:
    """Terminal node that composes and writes the final user-facing response."""
    llm_output, metric_final_output = await _llm_compose_user_facing_output(state)
    final_output = resolve_final_output(state, llm_output)

    return {
        "final_response": final_output,
        **build_metric_state_delta("final_output", "metric_final_output", metric_final_output),
    }
