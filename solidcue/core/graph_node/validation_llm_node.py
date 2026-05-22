from __future__ import annotations

import json
import re
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate
from solidcue.prompts.validation_llm_prompt import build_validation_messages

"""
Validation LLM Node - Function Overview
---------------------------------------

_fail:
Build standardized validation failure state delta.

_leaks_control_tokens:
Detect leaked internal control/tool-call token fragments.

_extract_json_object:
Parse validator raw output into JSON object safely.

_parse_validator_response:
Normalize validator output into `{passed, reason, score}`.

_llm_validate:
Run reviewer-model validation against evidence and draft output.

_artifact_has_delivery_id:
Check artifact payload for concrete delivery/upload identifiers.

validation_llm_node:
Main entrypoint. Phases:
1) Run deterministic pre-checks
2) Call LLM validator
3) Apply artifact-specific delivery checks
4) Emit final validation pass/fail + metrics
"""

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

def _fail(
    failure_type: str,
    reason: str,
    score: float = 0.0,
    *,
    metric_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "failure_type": failure_type,
        "validation_report": {"reason": reason, "score": score},
        **build_metric_state_delta("validation", "metric_validation", metric_validation or {}),
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
    return {"passed": passed, "reason": reason, "score": score}


def _llm_validate(state: AgentState, draft_output: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return None, {}

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "reviewer")
        context_evidence = state.get("context_evidence")
        evidence_for_validation: list[dict[str, Any]] = []
        if isinstance(context_evidence, list):
            role_tagged = [
                item
                for item in context_evidence
                if isinstance(item, dict)
                and item.get("evidence_role") in {"grounding", "alignment", "context"}
            ]
            if role_tagged:
                evidence_for_validation = [
                    item
                    for item in role_tagged
                    if item.get("evidence_role") in {"grounding", "alignment"}
                ]
            else:
                evidence_for_validation = [item for item in context_evidence if isinstance(item, dict)]
        task_description = ""
        task_plan = state.get("task_plan")
        current_task_id = state.get("current_task")
        if isinstance(task_plan, list) and current_task_id:
            current_task = next((t for t in task_plan if isinstance(t, dict) and t.get("id") == current_task_id), None)
            if current_task:
                task_description = str(current_task.get("description") or "")

        messages = build_validation_messages(
            user_query=str(state.get("user_input") or ""),
            draft_output=draft_output,
            context_evidence=evidence_for_validation,
            task_description=task_description,
        )
        raw_output, metric_validation = timed_generate(provider, messages)
        return _parse_validator_response(str(raw_output or "")), metric_validation
    except Exception:
        return None, {}


_ARTIFACT_ID_KEYS = frozenset({"documentId", "spreadsheetId", "id", "fileId", "webViewLink", "url", "filename"})


def _artifact_has_delivery_id(content: Any) -> bool:
    """Return True if the artifact tool response contains an ID or URL confirming delivery."""
    if not isinstance(content, dict):
        return False
    return any(content.get(k) for k in _ARTIFACT_ID_KEYS)


def validation_llm_node(state: AgentState) -> dict[str, Any]:
    """
    Validates synthesized user-facing draft content only.
    Emits only failure_type + validation_report. Router owns retry counters and
    dispatch; validation never routes.
    """
    draft_output = state.get("synthesis_draft") or state.get("draft_output")
    metric_validation: dict[str, Any] = {}

    if not isinstance(draft_output, str):
        return _fail("bad_synthesis", "Draft output must be a string.", metric_validation=metric_validation)

    draft_output = draft_output.strip()
    if not draft_output:
        return _fail("bad_synthesis", "Draft output is empty.", metric_validation=metric_validation)

    if _leaks_control_tokens(draft_output):
        return _fail(
            "bad_synthesis",
            "Draft output contains internal control tokens. Return only user-facing answer text.",
            metric_validation=metric_validation,
        )

    llm_result, metric_validation = _llm_validate(state, draft_output)
    if llm_result is None:
        return _fail(
            "bad_synthesis",
            "Validator could not produce a valid judgment. Regenerate a grounded answer.",
            metric_validation=metric_validation,
        )

    if llm_result.get("passed") is True:
        return {
            "failure_type": None,
            "validation_report": {"reason": llm_result.get("reason"), "score": llm_result.get("score")},
            **build_metric_state_delta("validation", "metric_validation", metric_validation),
        }

    reason = str(llm_result.get("reason") or "Validation failed.")
    return {
        "failure_type": "bad_synthesis",
        "validation_report": {"reason": reason, "score": llm_result.get("score")},
        **build_metric_state_delta("validation", "metric_validation", metric_validation),
    }
