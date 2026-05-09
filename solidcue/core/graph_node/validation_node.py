from __future__ import annotations

import json
import re
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.prompts.validation_prompt import build_validation_messages

CONTROL_TOKEN_FRAGMENTS = (
    "<|channel|>",
    "<|message|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "commentary<|channel|>",
    "analysis<|message|>",
)

ARTIFACT_REQUIRED_RETRY_PREFIX = "ARTIFACT_REQUIRED:"


def _validation_result(passed: bool, reason: str, score: float) -> dict[str, Any]:
    return {
        "passed": passed,
        "reason": reason,
        "score": score,
    }


def _fail_with_retry(state: AgentState, reason: str) -> dict[str, Any]:
    attempt_value = state.get("attempt")
    attempt = attempt_value if isinstance(attempt_value, int) else 0
    return {
        "validation_result": _validation_result(
            passed=False,
            reason=reason,
            score=0.0,
        ),
        "failure_type": "bad_synthesis",
        "validation_report": {"reason": reason},
        "router_origin": "validation",
        "retry_reason": reason,
        "attempt": attempt + 1,
    }


def _leaks_control_tokens(draft_output: str) -> bool:
    return any(fragment in draft_output for fragment in CONTROL_TOKEN_FRAGMENTS)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_validator_response(raw_output: str) -> dict[str, Any] | None:
    parsed = _extract_json_object(raw_output)
    if not isinstance(parsed, dict):
        return None

    passed = bool(parsed.get("passed", False))
    reason = str(parsed.get("reason", "Validator did not provide a reason.")).strip()
    score_raw = parsed.get("score", 0.0)
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    retry_tag_raw = str(parsed.get("retry_tag", "none")).strip().casefold()
    retry_tag = "artifact_required" if retry_tag_raw == "artifact_required" else "none"
    result = _validation_result(passed=passed, reason=reason, score=score)
    result["retry_tag"] = retry_tag
    return result


def _llm_validate(state: AgentState, draft_output: str) -> dict[str, Any] | None:
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return None

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "validator")
        messages = build_validation_messages(
            user_query=str(state.get("user_input") or ""),
            draft_output=draft_output,
            decision=state.get("decision") if isinstance(state.get("decision"), dict) else {},
            execution_result=(
                state.get("execution_result")
                if isinstance(state.get("execution_result"), dict)
                else {}
            ),
            tool_call_history=(
                state.get("tool_call_history")
                if isinstance(state.get("tool_call_history"), list)
                else []
            ),
            retry_reason=str(state.get("retry_reason") or ""),
            tool_turn_count=state.get("tool_turn_count")
            if isinstance(state.get("tool_turn_count"), int)
            else 0,
        )
        raw_output = provider.generate(messages)
        return _parse_validator_response(str(raw_output or ""))
    except Exception:
        return None


def validation_node(state: AgentState) -> dict[str, Any]:
    phase = state.get("phase")
    artifact_result = state.get("artifact_result")
    if phase == "artifact" and isinstance(artifact_result, dict):
        if artifact_result.get("success") is not True:
            reason = str(artifact_result.get("error") or "Artifact execution failed.")
            return {
                "validation_result": _validation_result(False, reason, 0.0),
                "failure_type": "not_executed",
                "validation_report": {"reason": reason},
                "router_origin": "validation",
                "retry_reason": reason,
            }
        draft_output = str(artifact_result.get("content") or "")
    else:
        draft_output = state.get("synthesis_draft") or state.get("draft_output")

    if not isinstance(draft_output, str):
        return _fail_with_retry(state, "Draft output must be a string.")

    draft_output = draft_output.strip()
    if not draft_output:
        return _fail_with_retry(state, "Draft output is empty.")

    if _leaks_control_tokens(draft_output):
        return _fail_with_retry(
            state,
            "Draft output contains internal control tokens. Return only user-facing answer text.",
        )

    llm_result = _llm_validate(state, draft_output)
    if llm_result is None:
        # Fail closed when validator output is unavailable/unparseable.
        return _fail_with_retry(
            state,
            "Validator could not produce a valid judgment. Regenerate a grounded answer.",
        )

    update: dict[str, Any] = {
        "validation_result": llm_result,
        "validation_report": {"reason": llm_result.get("reason"), "score": llm_result.get("score")},
        "router_origin": "validation",
    }
    if llm_result.get("passed") is True:
        update["finalization_reason"] = state.get("finalization_reason") or "validation_passed"
        update["failure_type"] = None
        return update

    reason = str(llm_result.get("reason") or "Validation failed.")
    failure_type = "bad_synthesis"
    if llm_result.get("retry_tag") == "artifact_required":
        if not reason.startswith(ARTIFACT_REQUIRED_RETRY_PREFIX):
            reason = f"{ARTIFACT_REQUIRED_RETRY_PREFIX} {reason}"
        if state.get("phase") == "artifact":
            failure_type = "bad_artifact"
        else:
            failure_type = "missing_source"
    attempt_value = state.get("attempt")
    attempt = attempt_value if isinstance(attempt_value, int) else 0
    update["failure_type"] = failure_type
    update["validation_report"] = {"reason": reason, "score": llm_result.get("score")}
    update["retry_reason"] = reason
    update["attempt"] = attempt + 1
    return update
