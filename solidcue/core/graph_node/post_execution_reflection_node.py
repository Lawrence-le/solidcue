import json
import logging
import re
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.prompts.reflection_prompt import build_reflection_messages

logger = logging.getLogger(__name__)

METADATA_ONLY_TOOL_TERMS = ("navigate", "status", "list", "search")
MATERIAL_CONTENT_KEYS = {
    "text",
    "content",
    "body",
    "markdown",
    "html",
    "data",
    "base64",
    "file_content",
}
METADATA_ONLY_CONTENT_KEYS = {
    "url",
    "title",
    "browser",
    "status",
    "ok",
    "error",
    "mimeType",
    "mime_type",
    "id",
    "name",
    "file_id",
    "documentId",
    "revisionId",
}


def _normalize_tool_input(tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return "{}"
    return json.dumps(tool_input, sort_keys=True, ensure_ascii=True, default=str)


def _tool_call_signature(state: AgentState) -> str | None:
    decision = state.get("decision")
    if not isinstance(decision, dict):
        return None
    tool_name = decision.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return None
    return f"{tool_name}:{_normalize_tool_input(decision.get('tool_input'))}"


def _current_tool_name(state: AgentState) -> str | None:
    decision = state.get("decision")
    if not isinstance(decision, dict):
        return None
    tool_name = decision.get("tool_name")
    return tool_name if isinstance(tool_name, str) and tool_name else None


def _is_metadata_only_content(content: Any) -> bool:
    if isinstance(content, dict):
        if any(key in content for key in MATERIAL_CONTENT_KEYS):
            return False
        keys = {str(key) for key in content}
        return bool(keys) and keys.issubset(METADATA_ONLY_CONTENT_KEYS)

    if isinstance(content, list):
        return bool(content) and all(_is_metadata_only_content(item) for item in content)

    return False


def _is_metadata_only_context(state: AgentState, content: Any) -> bool:
    if _is_metadata_only_content(content):
        return True

    tool_name = _current_tool_name(state)
    if not isinstance(tool_name, str):
        return False

    normalized_tool_name = tool_name.casefold()
    return (
        any(term in normalized_tool_name for term in METADATA_ONLY_TOOL_TERMS)
        and not isinstance(content, str)
        and _is_metadata_only_content(content)
    )


def _deterministic_reflection(state: AgentState) -> dict[str, Any] | None:
    """Handles immediate pass/fail based on raw execution health."""
    execution_result = state.get("execution_result")
    if not isinstance(execution_result, dict):
        return {
            "sufficient": False,
            "reason": "Execution context missing.",
            "missing": "execution_result",
        }

    if execution_result.get("success") is not True:
        return {
            "sufficient": False,
            "reason": "The latest tool execution failed.",
            "missing": "successful_result",
        }

    content = execution_result.get("content")
    if content is None or (isinstance(content, str) and not content.strip()):
        return {
            "sufficient": False,
            "reason": "Tool returned empty content.",
            "missing": "data",
        }

    if _is_metadata_only_context(state, content):
        return {
            "sufficient": False,
            "reason": "The latest tool returned metadata only, not source text.",
            "missing": "visible page text or source file content",
        }

    return None


def _robust_json_parser(raw_output: str) -> dict[str, Any] | None:
    """Attempts to extract JSON from markdown fences or conversational prose."""
    if not raw_output:
        return None

    cleaned = raw_output.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _parse_reflection_output(raw_output: str, state: AgentState) -> dict[str, Any]:
    """
    Parses LLM output into dict[str, Any].
    If parsing fails but evidence exists in state, it defaults to sufficient=True
    to prevent infinite search loops caused by formatting errors.
    """
    parsed = _robust_json_parser(raw_output)

    if isinstance(parsed, dict):
        return {
            "sufficient": bool(parsed.get("sufficient", False)),
            "reason": str(parsed.get("reason", "No reason provided.")).strip(),
            "missing": str(parsed.get("missing")).strip() if parsed.get("missing") else None,
        }

    execution_result = state.get("execution_result")
    if isinstance(execution_result, dict) and execution_result.get("content"):
        return {
            "sufficient": True,
            "reason": "Reflection formatting failed, but tool evidence is present.",
            "missing": "format_metadata",
        }

    return {
        "sufficient": False,
        "reason": "Unparsable sufficiency review and no clear tool evidence.",
        "missing": "valid_response",
    }


def _build_draft_from_execution(execution_result: Any, latest_output: Any) -> str:
    if isinstance(execution_result, dict):
        if execution_result.get("success") is True:
            content = execution_result.get("content")
            if content is not None:
                return str(content)
        error = execution_result.get("error")
        if error:
            return f"Tool execution failed: {error}"
    return str(latest_output or "")


def _force_context_stage_sufficient(state: AgentState, reflection: dict[str, Any]) -> dict[str, Any]:
    decision = state.get("decision")
    if not isinstance(decision, dict) or decision.get("tool_stage") != "context":
        return reflection

    execution_result = state.get("execution_result")
    if not isinstance(execution_result, dict):
        return reflection

    if execution_result.get("success") is not True:
        return reflection

    if _is_metadata_only_context(state, execution_result.get("content")):
        return reflection

    content = execution_result.get("content")
    if content is None or (isinstance(content, str) and not content.strip()):
        return reflection

    if reflection.get("sufficient") is True:
        return reflection

    return {
        "sufficient": True,
        "reason": "Context-stage tool execution succeeded and produced evidence; continue graph routing.",
        "missing": None,
    }


def post_execution_reflection_node(state: AgentState) -> dict[str, Any]:
    deterministic = _deterministic_reflection(state)
    if deterministic is not None:
        update: dict[str, Any] = {"reflection_result": deterministic}
    else:
        tool_call_history = state.get("tool_call_history", [])
        signature = _tool_call_signature(state)
        if signature:
            signature_count = sum(1 for call in tool_call_history if call.get("signature") == signature)
            if signature_count >= 2:
                update = {
                    "reflection_result": {
                        "sufficient": True,
                        "reason": "Maximum attempts for this specific tool reached. Proceeding with best evidence.",
                        "missing": None,
                    }
                }
            else:
                update = {}
        else:
            update = {}

    if not update:
        agent_key = state.get("agent_key")
        if not isinstance(agent_key, str) or not agent_key:
            update = {
                "reflection_result": {
                    "sufficient": False,
                    "reason": "Internal Error: Agent context lost.",
                    "missing": "agent_key",
                }
            }
        else:
            try:
                agent = load_agent(agent_key)
                provider = get_provider_for_role(agent, "sufficiency")

                messages = build_reflection_messages(
                    user_query=str(state.get("user_input", "")),
                    execution_result=state.get("execution_result"),
                    retry_reason=state.get("retry_reason"),
                    tool_stage=(
                        str(state.get("decision", {}).get("tool_stage"))
                        if isinstance(state.get("decision"), dict)
                        else None
                    ),
                )

                raw_response = provider.generate(messages)
                reflection = _parse_reflection_output(raw_response, state)
            except Exception as exc:
                logger.error("Reflection node exception: %s", exc)
                reflection = {
                    "sufficient": True if state.get("execution_result") else False,
                    "reason": "Reflection system encountered an error; using fallback logic.",
                    "missing": "system_exception",
                }
            reflection = _force_context_stage_sufficient(state, reflection)
            update = {"reflection_result": reflection}

    reflection = update.get("reflection_result")
    if not isinstance(reflection, dict):
        reflection = state.get("reflection_result", {})
    if isinstance(reflection, dict) and reflection.get("sufficient") is True:
        execution_result = update.get("execution_result", state.get("execution_result"))
        latest_output = update.get("latest_output", state.get("latest_output"))
        update["retry_reason"] = None
        update["draft_output"] = _build_draft_from_execution(execution_result, latest_output)
        update["finalization_reason"] = "reflection_sufficient"
        update["router_origin"] = "reflection"
        return update

    reflection_reason = reflection.get("reason") if isinstance(reflection, dict) else None
    reflection_missing = reflection.get("missing") if isinstance(reflection, dict) else None

    if isinstance(reflection_reason, str) and reflection_reason.strip():
        base_reason = reflection_reason.strip().rstrip(".")
    else:
        base_reason = "Post-execution evidence was insufficient."

    if isinstance(reflection_missing, str) and reflection_missing.strip():
        update["retry_reason"] = f"{base_reason}. Missing: {reflection_missing.strip().rstrip('.')}."
    else:
        update["retry_reason"] = base_reason

    update["router_origin"] = "reflection"
    return update
