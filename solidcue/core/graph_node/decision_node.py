from __future__ import annotations

import json
import re
import logging
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState, ToolCallState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate
from solidcue.prompts.decision_prompt import build_decision_messages
from solidcue.tools.loader import load_tool, get_missing_required_tool_fields, split_missing_tool_fields

logger = logging.getLogger(__name__)

"""
Decision Node - Function Overview
---------------------------------

DecisionValidator.validate:
Validate and normalize model decision payloads into ToolCallState.

DecisionValidator.as_fallback_response:
Build safe `respond` fallback decision when validation fails.

DecisionValidator._blocking_missing_required_fields:
Compute missing required tool fields that block execution.

_get_decision_payload:
Parse model output into a decision dict, including double-wrap recovery.

_extract_json_candidate:
Extract first valid decision-like JSON object from noisy text output.

_apply_neutral_decision_fallback:
Ensure respond decisions always contain a non-empty thought.

_resolve_available_tool_name:
Resolve/validate requested tool name against allowed tools.

decision_node:
Main orchestration entrypoint. Phases:
1) Build task-scoped prompt context
2) Resolve provider and run decision model
3) Parse output and validate/normalize ToolCallState
4) Persist active_tool_call and emit execution-ready state delta
"""


# --- Hardened Validation Layer ---

class DecisionValidator:
    """Runtime enforcement for the ToolCallState contract.

    Purpose:
    - Ensure the decision-model output is safe and executable before tool runtime.
    - Keep downstream nodes (`execution`, router) operating on a normalized shape.

    What it validates:
    - `action` is one of `use_tool` or `respond`.
    - `tool_name` resolves to an allowed tool for the active agent.
    - `tool_input` is a dict payload.
    - Required tool fields are present (blocking fields only).

    Behavior:
    - If validation passes, returns a normalized `use_tool` decision.
    - If validation fails, returns a safe `respond` fallback with an explanatory
      thought instead of executing a likely-failing tool call.

    Non-goals:
    - Does not infer or repair missing required fields.
    - Does not execute tools; it is a pre-execution guardrail only.
    """

    @staticmethod
    def validate(
        raw: dict[str, Any],
        available_tools: list[str],
    ) -> ToolCallState:
        action = raw.get("action")
        if action not in ["use_tool", "respond"]:
            action = "respond"

        if action == "use_tool":
            raw_name = raw.get("tool_name")
            tool_name = _resolve_available_tool_name(raw_name, available_tools)

            if not tool_name or tool_name not in available_tools:
                logger.warning("Rejecting invalid tool intent: %s", raw_name)
                return DecisionValidator.as_fallback_response(
                    "I couldn't safely execute that tool with the current agent configuration. "
                    "Please retry or choose a different request."
                )

            tool_input = raw.get("tool_input") if isinstance(raw.get("tool_input"), dict) else {}
            blocking_missing = DecisionValidator._blocking_missing_required_fields(tool_name, tool_input)
            if blocking_missing:
                fields_text = ", ".join(blocking_missing)
                logger.warning("Rejecting tool intent with missing required fields for %s: %s", tool_name, fields_text)
                return DecisionValidator.as_fallback_response(
                    f"I couldn't safely execute that tool because required inputs were missing "
                    f"({fields_text}). Please retry with complete details."
                )

            return {
                "action": "use_tool",
                "thought": str(raw.get("thought") or "Executing tool..."),
                "tool_name": tool_name,
                "tool_input": tool_input,
                "approval_preview": raw.get("approval_preview") if isinstance(raw.get("approval_preview"), dict) else None,
            }

        return {
            "action": "respond",
            "thought": str(raw.get("thought") or "Responding."),
            "tool_name": None,
            "tool_input": {},
            "approval_preview": None,
        }

    @staticmethod
    def as_fallback_response(message: str) -> ToolCallState:
        return {
            "action": "respond",
            "thought": message,
            "tool_name": None,
            "tool_input": {},
            "approval_preview": None,
        }

    @staticmethod
    def _blocking_missing_required_fields(
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> list[str]:
        try:
            tool = load_tool(tool_name)
        except Exception:
            return []
        missing = get_missing_required_tool_fields(tool, tool_input)
        _generatable, blocking = split_missing_tool_fields(missing)
        return blocking


# --- Primary Node Function ---

def decision_node(state: AgentState) -> dict[str, Any]:
    agent_key = state.get("agent_key")
    user_input = state.get("user_input", "")
    phase = state.get("phase") or "source"

    # 1. Build metadata with task context
    metadata = dict(state.get("metadata", {}))
    metadata["phase"] = phase
    task_plan = state.get("task_plan")
    current_task_id = state.get("current_task", "task_1")
    if isinstance(task_plan, list) and task_plan:
        current_task = next((t for t in task_plan if t.get("id") == current_task_id), None)
        if current_task:
            metadata["current_task"] = current_task
            metadata["current_task_id"] = current_task_id
            metadata["total_tasks"] = len(task_plan)

    tool_call_history = state.get("tool_call_history")

    # Show full context:
    # - All entries (success or fail) for current task → for retry context
    # - All successful entries from previous tasks → for output references (IDs, payloads, etc.)
    scoped_tool_call_history = (
        [
            entry
            for entry in tool_call_history
            if isinstance(entry, dict) and (
                entry.get("task_id") == current_task_id or  # All current task attempts
                entry.get("success") is True                # All previous successes
            )
        ]
        if isinstance(tool_call_history, list)
        else None
    )

    # 2. Call LLM
    agent_config = load_agent(agent_key)
    messages = build_decision_messages(
        agent=agent_config,
        user_input=user_input,
        retry_reason=state.get("retry_reason"),
        metadata=metadata,
        tool_call_history=scoped_tool_call_history,
    )
    provider = get_provider_for_role(agent_config, "brain")
    output_text, metric_decision = timed_generate(provider, messages)

    available_tools = agent_config.tools if hasattr(agent_config, "tools") else []
    if not available_tools:
        logger.error("Critical: Agent %s loaded without tool configuration.", agent_key)

    # 3. Parse and validate
    raw_payload = _get_decision_payload(output_text)
    decision = DecisionValidator.validate(raw_payload, available_tools)
    decision = _apply_neutral_decision_fallback(decision, output_text)
    tool_use = decision["action"] == "use_tool"

    # 4. Return state delta
    update: dict[str, Any] = {
        "decision": decision,
        "tool_use": tool_use,
        "llm_prompt_messages": messages,
        **build_metric_state_delta("decision", "metric_decision", metric_decision),
    }

    if tool_use:
        update["active_tool_call"] = decision

    return update


# --- Parsing ---

def _get_decision_payload(output_text: str) -> dict[str, Any]:
    """Parse LLM output into a valid decision payload.

    Extracts and validates a decision JSON object from raw LLM output, handling
    edge cases where the LLM generates the correct decision but wraps it incorrectly.

    Process:
    1. Extract JSON from output using _extract_json_candidate
    2. Parse and validate it contains an "action" key
    3. Handle double-wrapped case: if outer action is "respond" but the "thought"
       field contains a use_tool JSON decision, unwrap and return the inner decision
    4. Return the valid decision, or fallback to {"action": "respond", ...}

    The double-wrap case occurs when LLMs generate correct tool decisions but embed
    them in the thought field of a respond action instead of at the top level.

    Returns: A decision dict with "action" (use_tool or respond), "tool_name", "tool_input", etc.
    """
    json_str = _extract_json_candidate(output_text)
    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "action" in data:
            thought = data.get("thought")
            if isinstance(thought, str) and data.get("action") == "respond":
                try:
                    inner_json = _extract_json_candidate(thought)
                    inner = json.loads(inner_json)
                    if isinstance(inner, dict) and inner.get("action") == "use_tool":
                        return inner
                except Exception:
                    pass
            return data
    except Exception:
        pass
    return {"action": "respond", "thought": output_text}


def _extract_json_candidate(text: str) -> str:
    """Extract the first valid JSON object from text, handling various LLM output formats.

    LLMs don't always return clean JSON — they may include markdown fences, prose,
    or multiple JSON objects concatenated. This function locates the first valid
    JSON object that contains an "action" key (the decision payload).

    Handles:
    - Markdown code fences: ```json {...} ```
    - JSON starting at text beginning: {...}...
    - JSON buried in prose: some text {...} more text

    Returns the first valid JSON object as a string, or the original text if
    no valid JSON with "action" key is found.
    """
    stripped = text.strip()

    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()

    decoder = json.JSONDecoder()

    if stripped.startswith("{"):
        try:
            candidate, end = decoder.raw_decode(stripped)
            if isinstance(candidate, dict) and "action" in candidate:
                return json.dumps(candidate)
        except json.JSONDecodeError:
            pass
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "action" in candidate:
            return json.dumps(candidate)

    return stripped


# --- Helpers ---

def _apply_neutral_decision_fallback(decision: ToolCallState, output_text: str) -> ToolCallState:
    if decision.get("action") != "respond":
        return decision
    if str(decision.get("thought") or "").strip():
        return decision
    updated = dict(decision)
    updated["thought"] = str(output_text or "").strip() or "Decision fallback: no valid JSON output."
    return updated


def _resolve_available_tool_name(tool_name: Any, available_tools: list[str]) -> str | None:
    if not isinstance(tool_name, str):
        return None
    if tool_name in available_tools:
        return tool_name
    return None
