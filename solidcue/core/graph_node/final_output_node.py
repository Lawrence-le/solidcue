import json
import re
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate
from solidcue.prompts.final_output_prompt import build_final_output_messages


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


def _llm_compose_user_facing_output(state: AgentState) -> tuple[str | None, dict[str, Any]]:
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return None, {}

    execution_result = state.get("execution_result")
    if not isinstance(execution_result, dict):
        return None, {}

    last_tool_call = state.get("active_tool_call")
    if not isinstance(last_tool_call, dict):
        decision = state.get("decision")
        last_tool_call = decision if isinstance(decision, dict) else {}

    payload = {
        "user_input": str(state.get("user_input") or ""),
        "tool_name": last_tool_call.get("tool_name"),
        "tool_input": last_tool_call.get("tool_input") if isinstance(last_tool_call.get("tool_input"), dict) else {},
        "execution_result": execution_result,
    }

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "lite")
        messages = build_final_output_messages(payload)
        output, metric_final_output = timed_generate(provider, messages)
        if isinstance(output, str) and output.strip():
            return output.strip(), metric_final_output
    except Exception:
        return None, {}
    return None, {}


def final_output_node(state: AgentState) -> dict[str, Any]:
    """
    Terminal node. Reads synthesis_draft (normal path) or falls back to
    fallback logic. Writes only final_response.
    """
    if state.get("phase") == "conversational":
        return {
            "final_response": state.get("final_response") or _build_fallback_output(state),
        }

    llm_output, metric_final_output = _llm_compose_user_facing_output(state)
    final_output = (
        llm_output
        or state.get("synthesis_draft")
        or _build_fallback_output(state)
    )

    return {
        "final_response": final_output,
        **build_metric_state_delta("final_output", "metric_final_output", metric_final_output),
    }
