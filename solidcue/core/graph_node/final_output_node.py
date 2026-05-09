import json
import re
from typing import Any

from solidcue.core.state.schema import AgentState


_SENSITIVE_QUERY_PARAM_RE = re.compile(r"([?&](?:api_key|key|token|access_token)=)[^&\s]+", re.IGNORECASE)


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


def final_output_node(state: AgentState) -> dict[str, Any]:
    final_output = (
        state.get("final_response")
        or state.get("synthesis_draft")
        or state.get("draft_output")
        or _build_fallback_output(state)
    )

    return {
        "final_output": final_output,
        "final_response": final_output,
        "workflow_status": "completed",
    }
