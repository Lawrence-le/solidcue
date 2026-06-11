from __future__ import annotations

import json
import logging
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.graph_agent.state.schema import AgentState
from solidcue.core.graph_agent.nodes.validation_llm_node import _artifact_has_delivery_id

logger = logging.getLogger(__name__)

"""
Router Node - Function Overview
-------------------------------

_get_latest_execution_result:
Read latest execution result from `tool_call_history` (fallback to state field).

_guardrail_is_artifact_intent:
Detect artifact-oriented user intent from input keywords.

_guardrail_retry_limit_reached:
Enforce global retry-attempt budget across phases.

_guardrail_artifact_required_retry:
Detect validator-enforced artifact retry from validation report.

_guardrail_next_artifact_task_id:
Find next artifact task id at/after current task pointer.

_guardrail_latest_task_failure_detail:
Extract latest failure detail for the current task from history.

_guardrail_tools_tried_for_task:
Collect unique tools already attempted for a specific task.

_guardrail_build_retry_reason:
Build structured retry reason text for artifact retries.

_guardrail_task_position_label:
Render human-readable "task X of Y" position label.

_check_task_completion_by_accomplishments:
Determine whether a task is complete by accomplishment tags and execution status.

_guardrail_phase_for_task_type:
Map task type -> phase name.

_guardrail_router_next_for_task_type:
Map task type -> next node for router.

_guardrail_advance_task_plan:
Advance task pointer and compute phase/router transitions safely.

router_node:
Main entrypoint. Phases:
1) Evaluate guardrails/retry constraints
2) Advance or retain current task state
3) Route to next node with phase metadata
"""

ARTIFACT_INTENT_KEYWORDS = (
    "document",
    "google doc",
    "pdf",
    "word document",
    "spreadsheet",
    "tracker",
    "save file",
    "upload file",
    "write",
    "generate",
    "create",
)


def _get_latest_execution_result(state: AgentState) -> dict[str, Any] | None:
    """Extract the latest execution result from tool_call_history.

    Single source of truth: tool_call_history contains all tool calls and their results.
    Falls back to state["execution_result"] for backward compatibility.
    """
    history = state.get("tool_call_history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict):
            result = latest.get("execution_result")
            if result is not None:
                return result
    return state.get("execution_result")


def _guardrail_is_artifact_intent(user_input: Any) -> bool:
    if not isinstance(user_input, str):
        return False
    normalized = user_input.casefold()
    return any(keyword in normalized for keyword in ARTIFACT_INTENT_KEYWORDS)


def _guardrail_retry_limit_reached(state: AgentState) -> bool:
    max_retries = state.get("max_retries") if isinstance(state.get("max_retries"), int) else 10
    total = sum(
        value if isinstance(value, int) else 0
        for value in (
            state.get("source_attempt"),
            state.get("artifact_attempt"),
            state.get("synthesis_attempt"),
        )
    )
    return total >= max_retries


def _guardrail_artifact_required_retry(state: AgentState) -> bool:
    validation_report = state.get("validation_report")
    if not isinstance(validation_report, dict):
        return False
    reason = str(validation_report.get("reason") or "")
    return reason.startswith("ARTIFACT_REQUIRED:")


def _guardrail_next_artifact_task_id(state: AgentState) -> str | None:
    """Return the id of the next artifact_generation task at or after current_task."""
    task_plan = state.get("task_plan")
    current_task_id = state.get("current_task", "task_1")
    if not isinstance(task_plan, list):
        return None

    current_idx = next(
        (i for i, t in enumerate(task_plan) if isinstance(t, dict) and t.get("id") == current_task_id),
        0,
    )
    for task in task_plan[current_idx:]:
        if isinstance(task, dict) and task.get("type") == "artifact_generation":
            return str(task.get("id"))
    return None


def _guardrail_latest_task_failure_detail(state: AgentState, task_id: str) -> str:
    """Extract a concise failure detail from the latest failed current-task tool call."""
    history = state.get("tool_call_history")
    if not isinstance(history, list):
        return ""

    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if entry.get("task_id") != task_id:
            continue
        if entry.get("success") is not False:
            continue

        execution_result = entry.get("execution_result")
        if isinstance(execution_result, dict):
            error = execution_result.get("error")
            if isinstance(error, str) and error.strip():
                detail = error.strip()
                return detail[:300] + ("…" if len(detail) > 300 else "")

        if isinstance(execution_result, dict):
            content = execution_result.get("content")
            if isinstance(content, str) and content.strip():
                detail = content.strip()
                return detail[:300] + ("…" if len(detail) > 300 else "")
            if content is not None:
                detail = json.dumps(content, ensure_ascii=True, default=str).strip()
                if detail:
                    return detail[:300] + ("…" if len(detail) > 300 else "")
        return ""
    return ""


def _guardrail_tools_tried_for_task(state: AgentState, task_id: str) -> list[str]:
    """Return list of tool names already called for the given task_id."""
    history = state.get("tool_call_history")
    if not isinstance(history, list):
        return []
    tools: list[str] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("task_id") != task_id:
            continue
        name = entry.get("tool_name")
        if isinstance(name, str) and name and name not in tools:
            tools.append(name)
    return tools


def _guardrail_build_retry_reason(
    satisfied: list[str],
    missing: list[str],
    incomplete_reason: str,
    task_position: str = "",
    planned_tool: str | None = None,
    tools_tried: list[str] | None = None,
    attempt: int = 0,
) -> str:
    """Build an actionable retry_reason for decision guidance.

    Includes:
    - What's missing
    - What tool should be used (per task plan)
    - What tools were already tried (to avoid repetition)
    - Explicit action instruction
    """
    target = ",".join(missing[:3]) if missing else "current_task_requirements"
    action = incomplete_reason.strip() if incomplete_reason else "Collect missing requirements using the correct tool."

    lines = [
        f"MISSING: {target}",
        f"REASON: {action}",
    ]

    # Add planned tool instruction (CRITICAL for decision)
    if isinstance(planned_tool, str) and planned_tool.strip():
        lines.append(f"REQUIRED_TOOL: {planned_tool.strip()} (use this exact tool, no substitutes)")

    # Show what's been tried to avoid repetition
    if tools_tried:
        unique_tried = ", ".join(tools_tried[:5])
        lines.append(f"ALREADY_TRIED: {unique_tried}")
        if planned_tool and planned_tool not in tools_tried:
            lines.append(f"NEXT_ACTION: Switch to {planned_tool}. Previous attempts used wrong tools.")
        elif planned_tool and planned_tool in tools_tried and attempt >= 2:
            lines.append(f"NEXT_ACTION: {planned_tool} was tried but failed/incomplete. Check inputs from previous task outputs in tool history.")

    if task_position:
        lines.append(f"TASK: {task_position}")

    if attempt >= 3:
        lines.append(f"WARNING: This is retry #{attempt}. If you cannot make progress, respond with action=respond and explain the blocker.")

    lines.append("CONSTRAINT: Only work on the current task. Do NOT call tools from other tasks.")
    return "\n".join(lines)


def _guardrail_task_position_label(task_plan: list[dict[str, Any]], current_task_id: str) -> str:
    """Return a human-readable label like 'task 2 of 4 (source_gathering)'."""
    total = len(task_plan)
    for idx, t in enumerate(task_plan, start=1):
        if isinstance(t, dict) and t.get("id") == current_task_id:
            task_type = str(t.get("type") or "unknown")
            description = str(t.get("description") or "")
            label = f"You are on task {idx} of {total} ({task_type})."
            if description:
                label += f" Goal: {description}"
            return label
    return ""


def _check_task_completion_by_accomplishments(state: AgentState, task: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Check task completion deterministically using accomplishments from tool_call_history.

    Reflection node writes accomplishments with "_met" or "_missing" suffix.
    Task is complete when all requires have a corresponding "_met" accomplishment.

    Returns (complete, satisfied, missing).
    """
    task_id = str(task.get("id") or "")
    requires = task.get("requires")
    if not isinstance(requires, list):
        return True, [], []

    required_set = set(r for r in requires if isinstance(r, str) and r.strip())
    if not required_set:
        return True, [], []

    # Collect accomplishments from successful tool calls in this task
    history = state.get("tool_call_history")
    met_set: set[str] = set()
    missing_set: set[str] = set()

    if isinstance(history, list):
        for entry in history:
            if not isinstance(entry, dict):
                continue
            if entry.get("task_id") != task_id:
                continue
            if entry.get("success") is not True:
                continue
            accomplishments = entry.get("accomplishments")
            if isinstance(accomplishments, list):
                for acc in accomplishments:
                    if not isinstance(acc, str):
                        continue
                    acc_clean = acc.strip()
                    if acc_clean.endswith("_met"):
                        req = acc_clean[:-4]  # Remove "_met" suffix
                        if req in required_set:
                            met_set.add(req)
                    elif acc_clean.endswith("_missing"):
                        req = acc_clean[:-8]  # Remove "_missing" suffix
                        if req in required_set:
                            missing_set.add(req)

    # Task complete if all requires are met
    satisfied = sorted(list(met_set))
    missing = sorted(list(required_set - met_set))
    task_complete = len(missing) == 0

    logger.debug(
        "router: task %s accomplishment check: required=%s met=%s missing=%s incomplete=%s",
        task_id, sorted(list(required_set)), satisfied, sorted(list(missing_set)), missing, task_complete
    )

    return task_complete, satisfied, missing


def _guardrail_phase_for_task_type(task_type: str) -> str:
    normalized = str(task_type or "").strip().casefold()
    if normalized == "source_gathering":
        return "source"
    if normalized == "artifact_generation":
        return "artifact"
    return "synthesis"


def _guardrail_router_next_for_task_type(task_type: str) -> str:
    normalized = str(task_type or "").strip().casefold()
    if normalized in {"source_gathering", "artifact_generation"}:
        return "decision"
    return "synthesis"


def _guardrail_advance_task_plan(state: AgentState) -> dict[str, Any]:
    """Advance task plan only when current task completion evidence exists."""
    task_plan = state.get("task_plan")
    current_task_id = state.get("current_task", "task_1")

    if not isinstance(task_plan, list) or len(task_plan) == 0:
        return {}

    # Find current task by id
    current_idx = next(
        (i for i, t in enumerate(task_plan) if isinstance(t, dict) and t.get("id") == current_task_id),
        None,
    )
    if current_idx is None:
        return {}

    current_task = task_plan[current_idx]
    current_task_type = str(current_task.get("type") or "synthesis")
    current_task_status = str(current_task.get("status") or "pending").casefold()
    current_phase = _guardrail_phase_for_task_type(current_task_type)
    current_next = _guardrail_router_next_for_task_type(current_task_type)

    # Route within current task until it has concrete completion evidence.
    if current_task_type in {"source_gathering", "artifact_generation"}:
        if current_task_status == "completed":
            task_complete = True
            satisfied = list(current_task.get("requires") or [])
            missing: list[str] = []
        else:
            # Use deterministic accomplishment checking instead of LLM task completion
            task_complete, satisfied, missing = _check_task_completion_by_accomplishments(state, current_task)
        if not task_complete:
            task_plan_list = task_plan if isinstance(task_plan, list) else []
            # Build action hint from missing accomplishments
            action_hint = ""
            if missing:
                action_hint = f"Task output did not satisfy: {', '.join(missing[:2])}"
            task_context = current_task.get("context") if isinstance(current_task.get("context"), dict) else {}
            planned_tool = task_context.get("tool") if isinstance(task_context, dict) else None
            tools_tried = _guardrail_tools_tried_for_task(state, current_task_id)
            attempts_for_task = len(tools_tried)
            return {
                "phase": current_phase,
                "current_task": current_task_id,
                "router_next": current_next,
                "retry_reason": _guardrail_build_retry_reason(
                    satisfied,
                    missing,
                    action_hint,
                    task_position=_guardrail_task_position_label(task_plan_list, current_task_id),
                    planned_tool=planned_tool,
                    tools_tried=tools_tried,
                    attempt=attempts_for_task,
                ),
            }
    elif current_task_type in {"synthesis", "review"}:
        if current_task_status != "completed":
            draft = state.get("synthesis_draft")
            failure_type = state.get("failure_type")
            # Validation passed and draft exists — mark complete and advance
            if not (isinstance(draft, str) and draft.strip() and failure_type is None):
                return {
                    "phase": "synthesis",
                    "current_task": current_task_id,
                    "router_next": "synthesis",
                    "retry_reason": (state.get("validation_report") or {}).get("reason"),
                }
    elif current_task_type not in {"source_gathering", "artifact_generation", "synthesis", "review"}:
        return {"phase": "synthesis", "current_task": current_task_id, "router_next": "synthesis"}

    # Current task complete — mark it as completed in task_plan
    updated_task_plan = [
        {**t, "status": "completed"} if t.get("id") == current_task_id else t
        for t in task_plan
    ]

    next_idx = current_idx + 1

    if next_idx >= len(task_plan):
        if current_task_type in {"artifact_generation", "synthesis", "review"}:
            return {"phase": "final", "task_plan": updated_task_plan, "retry_reason": None, "router_next": "final_output"}
        if _guardrail_is_artifact_intent(state.get("user_input")):
            return {"phase": "artifact", "task_plan": updated_task_plan, "retry_reason": None, "router_next": "decision"}
        return {"phase": "synthesis", "task_plan": updated_task_plan, "retry_reason": None, "router_next": "synthesis"}

    next_task = task_plan[next_idx]
    next_task_id = str(next_task.get("id", f"task_{next_idx + 1}"))
    next_task_type = next_task.get("type", "synthesis")
    next_phase = _guardrail_phase_for_task_type(str(next_task_type))
    next_router_next = _guardrail_router_next_for_task_type(str(next_task_type))
    update: dict[str, Any] = {
        "phase": next_phase,
        "current_task": next_task_id,
        "task_plan": updated_task_plan,
        "retry_reason": None,
        "router_next": next_router_next,
    }
    if next_phase == "synthesis":
        update["synthesis_draft"] = None
    return update


def router_node(state: AgentState) -> dict[str, Any]:
    """
    Sole router for the agent graph. Dispatches uniformly on `failure_type`
    and `phase`. Checks task completion deterministically via accomplishments
    from tool_call_history, enabling source-type agnostic routing.
    """
    if _guardrail_retry_limit_reached(state):
        return {"phase": "final", "failure_type": "retry_limit", "router_next": "final_output"}

    phase = state.get("phase") or "source"
    failure_type = state.get("failure_type")

    if failure_type == "missing_source":
        if _guardrail_artifact_required_retry(state) and (
            _guardrail_next_artifact_task_id(state) is not None or _guardrail_is_artifact_intent(state.get("user_input"))
        ):
            next_artifact_id = _guardrail_next_artifact_task_id(state)
            route: dict[str, Any] = {
                "phase": "artifact",
                "failure_type": None,
                "router_next": "decision",
            }
            if next_artifact_id is not None:
                route["current_task"] = next_artifact_id
            return route

        source_attempt = state.get("source_attempt") if isinstance(state.get("source_attempt"), int) else 0
        current_task_id = state.get("current_task", "task_1")
        task_plan = state.get("task_plan")
        current_task = {}
        task_plan_list: list[dict[str, Any]] = []
        if isinstance(task_plan, list):
            task_plan_list = task_plan
            current_task = next((t for t in task_plan if t.get("id") == current_task_id), {}) or {}
        failure_detail = _guardrail_latest_task_failure_detail(state, current_task_id)
        incomplete_reason = (
            f"Previous tool call failed: {failure_detail}"
            if failure_detail
            else "Previous tool call failed."
        )
        task_context = current_task.get("context") if isinstance(current_task.get("context"), dict) else {}
        planned_tool = task_context.get("tool") if isinstance(task_context, dict) else None
        tools_tried = _guardrail_tools_tried_for_task(state, current_task_id)
        return {
            "phase": "source",
            "current_task": current_task_id,
            "source_attempt": source_attempt + 1,
            "retry_reason": _guardrail_build_retry_reason(
                [],
                list(current_task.get("requires") or []),
                incomplete_reason,
                task_position=_guardrail_task_position_label(task_plan_list, current_task_id),
                planned_tool=planned_tool,
                tools_tried=tools_tried,
                attempt=source_attempt + 1,
            ),
            "router_next": "decision",
        }

    if failure_type == "bad_artifact":
        artifact_attempt = state.get("artifact_attempt") if isinstance(state.get("artifact_attempt"), int) else 0
        current_task_id = state.get("current_task", "task_1")
        task_plan = state.get("task_plan")
        current_task = {}
        task_plan_list: list[dict[str, Any]] = []
        if isinstance(task_plan, list):
            task_plan_list = task_plan
            current_task = next((t for t in task_plan if t.get("id") == current_task_id), {}) or {}
        failure_detail = _guardrail_latest_task_failure_detail(state, current_task_id)
        incomplete_reason = (
            f"Artifact tool execution failed: {failure_detail}"
            if failure_detail
            else "Artifact tool execution failed. Check tool inputs and retry."
        )
        task_context = current_task.get("context") if isinstance(current_task.get("context"), dict) else {}
        planned_tool = task_context.get("tool") if isinstance(task_context, dict) else None
        tools_tried = _guardrail_tools_tried_for_task(state, current_task_id)
        return {
            "phase": "artifact",
            "current_task": current_task_id,
            "artifact_attempt": artifact_attempt + 1,
            "retry_reason": _guardrail_build_retry_reason(
                [],
                list(current_task.get("requires") or []),
                incomplete_reason,
                task_position=_guardrail_task_position_label(task_plan_list, current_task_id),
                planned_tool=planned_tool,
                tools_tried=tools_tried,
                attempt=artifact_attempt + 1,
            ),
            "router_next": "decision",
        }

    if failure_type == "not_executed":
        artifact_attempt = state.get("artifact_attempt") if isinstance(state.get("artifact_attempt"), int) else 0
        return {
            "phase": "artifact",
            "artifact_attempt": artifact_attempt + 1,
            "router_next": "decision",
        }

    if failure_type == "bad_synthesis":
        synthesis_attempt = state.get("synthesis_attempt") if isinstance(state.get("synthesis_attempt"), int) else 0
        validation_report = state.get("validation_report")
        retry_reason = (
            validation_report.get("reason")
            if isinstance(validation_report, dict)
            else None
        )
        return {
            "phase": "synthesis",
            "synthesis_attempt": synthesis_attempt + 1,
            "retry_reason": retry_reason,
            "router_next": "synthesis",
        }

    # failure_type is None — handle phase transitions
    task_advance = _guardrail_advance_task_plan(state)
    if task_advance:
        return task_advance

    # Fallback to phase-based routing
    if phase == "source":
        if _guardrail_is_artifact_intent(state.get("user_input")):
            return {"phase": "artifact", "failure_type": None, "router_next": "decision"}
        return {"phase": "synthesis", "failure_type": None, "router_next": "synthesis"}

    if phase == "artifact":
        execution_result = _get_latest_execution_result(state)
        decision = state.get("decision")
        decision_is_tool_call = (
            isinstance(decision, dict)
            and decision.get("action") == "use_tool"
            and isinstance(decision.get("tool_name"), str)
            and bool(str(decision.get("tool_name")).strip())
        )
        if not (
            decision_is_tool_call
            and
            isinstance(execution_result, dict)
            and execution_result.get("success") is True
            and _artifact_has_delivery_id(execution_result.get("content"))
        ):
            return {"phase": "artifact", "router_next": "decision"}
        return {"phase": "synthesis", "router_next": "synthesis"}

    if phase == "synthesis":
        return {"phase": "final", "router_next": "final_output"}

    return {"phase": "final", "router_next": "final_output"}
