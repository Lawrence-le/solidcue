from __future__ import annotations

import json
import re
from typing import Any

from solidcue.agent_configs.loader import load_agent
from solidcue.providers.provider_resolver import get_provider_for_role
from solidcue.core.graph_agent.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate
from solidcue.core.graph_agent.prompts.validation_llm_prompt import build_validation_messages

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

_current_task_item_key:
Resolve current task item scope key for multi-item evidence filtering.

_handoff_for_item / _build_validation_evidence_from_handoff:
Build bounded, item-scoped validation evidence from handoff payloads.

_llm_validate:
Run reviewer-model validation against item-scoped handoff evidence and draft output.

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
_VALIDATION_EVIDENCE_MAX_CHARS = 12000
_VALIDATION_ENTRY_MAX_CHARS = 3000
_TEXT_FIELD_CANDIDATES = ("content", "text", "body", "markdown")

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


def _current_task_item_key(state: AgentState) -> str | None:
    task_plan = state.get("task_plan")
    current_task_id = state.get("current_task")
    if not isinstance(task_plan, list) or not current_task_id:
        return None
    current_task = next((t for t in task_plan if isinstance(t, dict) and t.get("id") == current_task_id), None)
    if not isinstance(current_task, dict):
        return None
    context = current_task.get("context")
    if not isinstance(context, dict):
        return None
    item_key = context.get("item_key")
    if not isinstance(item_key, str):
        return None
    cleaned = item_key.strip()
    return cleaned or None


def _handoff_for_item(handoff: dict[str, Any], item_key: str | None) -> dict[str, Any]:
    """Build validation handoff view for current item + global shared entries."""
    scoped: dict[str, Any] = {}
    if item_key:
        suffix = f"::{item_key}"
        for key, value in handoff.items():
            if not isinstance(key, str):
                continue
            if key.endswith(suffix):
                scoped[key[: -len(suffix)]] = value

    for key, value in handoff.items():
        if isinstance(key, str) and key.startswith("global::"):
            scoped[key[len("global::"):]] = value

    return scoped


def _extract_text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in _TEXT_FIELD_CANDIDATES:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for nested in value.values():
            text = _extract_text_from_value(nested)
            if text:
                return text
    if isinstance(value, list):
        for item in value:
            text = _extract_text_from_value(item)
            if text:
                return text
    return ""


def _build_validation_evidence_from_handoff(state: AgentState) -> list[dict[str, Any]]:
    """Create bounded validation evidence from item-scoped handoff payloads."""
    handoff = state.get("handoff")
    if not isinstance(handoff, dict) or not handoff:
        return []

    item_key = _current_task_item_key(state)
    handoff_view = _handoff_for_item(handoff, item_key)
    evidence: list[dict[str, Any]] = []
    total_chars = 0

    for source_key, value in handoff_view.items():
        if not isinstance(source_key, str):
            continue
        if "base64" in source_key.casefold():
            continue

        text = _extract_text_from_value(value)
        if not text:
            continue
        if len(text) > _VALIDATION_ENTRY_MAX_CHARS:
            text = text[:_VALIDATION_ENTRY_MAX_CHARS].rstrip() + "… [truncated]"
        remaining = _VALIDATION_EVIDENCE_MAX_CHARS - total_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "… [truncated]"
        evidence.append(
            {
                "source_key": source_key,
                "item_key": item_key or "",
                "content": text,
            }
        )
        total_chars += len(text)

    return evidence


def _llm_validate(state: AgentState, draft_output: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return None, {}

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "reviewer")
        evidence_for_validation = _build_validation_evidence_from_handoff(state)
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
            validation_evidence=evidence_for_validation,
            task_description=task_description,
        )
        raw_output, metric_validation = timed_generate(
            provider,
            messages,
            node_name="validation",
        )
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
    draft_output = state.get("synthesis_draft")
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
