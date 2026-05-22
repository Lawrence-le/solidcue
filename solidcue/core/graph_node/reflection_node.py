import json
import logging
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric, build_metric_state_delta, timed_generate
from solidcue.prompts.reflection_prompt import build_reflection_messages

logger = logging.getLogger(__name__)

"""
Reflection Node - Function Overview
-----------------------------------

_llm_check_requires_met:
LLM-based semantic check for requirement satisfaction.

_check_requires_satisfied:
Hybrid requirement checker (deterministic pass + optional LLM fallback).

_update_tool_call_history_with_accomplishments:
Attach met/missing accomplishments to the latest task history entry.

_merge_token_stats:
Merge token usage/time/model stats from sub-steps.

_evidence_signature / _append_context_evidence:
Deduplicate and append evidence entries into `context_evidence`.

_has_substantial_text / _effective_evidence_role:
Determine whether content is useful and classify evidence role.

Main entrypoint:
- `reflection_node`: resolve task requires, evaluate met/missing, update state.
"""


def _llm_check_requires_met(
    requires: list[str],
    execution_result: Any,
    agent_key: str,
    tool_name: str = "unknown",
) -> tuple[list[str], dict[str, Any]]:
    """Use LLM to check if each requirement is satisfied by the tool execution result."""
    if not requires:
        return [], {}

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "lite")

        # Format execution result for display (includes success, error, content)
        result_dict = {
            "success": execution_result.get("success") if isinstance(execution_result, dict) else False,
            "error": execution_result.get("error") if isinstance(execution_result, dict) else None,
            "content": execution_result.get("content") if isinstance(execution_result, dict) else execution_result,
        }
        result_str = json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)[:4000]

        if not result_str.strip() or result_str == "{}":
            return [f"{req.strip()}_missing" for req in requires if isinstance(req, str) and req.strip()], {}

        messages = build_reflection_messages(
            tool_name=tool_name,
            requires=requires,
            execution_result=result_str,
        )
        response, metric = timed_generate(provider, messages)
        token_stats = dict(metric.get("tokens") or {})
        token_stats["time_s"] = float(metric.get("time_s") or 0.0)
        token_stats["model"] = str(metric.get("model") or "")

        # Parse response
        try:
            result = json.loads(response)
            if not isinstance(result, dict):
                result = {}
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r"\{[^}]+\}", response)
            if match:
                try:
                    result = json.loads(match.group(0))
                except json.JSONDecodeError:
                    result = {}
            else:
                result = {}

        # Build accomplishments based on LLM response
        accomplishments = []
        for req in requires:
            if not isinstance(req, str) or not req.strip():
                continue
            req_clean = req.strip()

            # Check if LLM said this requirement is met
            is_met = result.get(req_clean) or result.get(req_clean.lower())
            if is_met is True:
                accomplishments.append(f"{req_clean}_met")
            else:
                accomplishments.append(f"{req_clean}_missing")

        return accomplishments, token_stats
    except Exception as exc:
        logger.exception("reflection: LLM requirement check failed: %s", exc)
        # Fallback: if content exists, assume requirements are met
        return [f"{req.strip()}_met" for req in requires if isinstance(req, str) and req.strip()], {}


def _check_requires_satisfied(
    requires: list[str],
    execution_result: Any,
    agent_key: str = None,
    tool_name: str = "unknown",
    planned_tool: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Check which required accomplishments are satisfied by the tool execution result.

    Decision logic:
    1. If execution failed → all requires _missing (deterministic)
    2. If planned tool was used AND execution succeeded → all requires _met (deterministic, no LLM)
       Rationale: planning chose the tool to satisfy these requires. If it ran successfully,
       the task is done. LLM second-guessing creates infinite loops when requirement labels
       are aspirational (e.g., 'next_empty_row_identified' vs read tool output).
    3. Otherwise (different tool or no planned tool) → LLM validates semantically

    Returns list of accomplishments with "_met" or "_missing" suffix.
    """
    if not isinstance(requires, list):
        return [], {}

    accomplishments: list[str] = []

    # 1. If execution failed, all requires are missing
    if not isinstance(execution_result, dict) or execution_result.get("success") is not True:
        for req in requires:
            if isinstance(req, str) and req.strip():
                accomplishments.append(f"{req.strip()}_missing")
        return accomplishments, {}

    # 2. If planned tool matches executed tool, trust the plan (skip LLM)
    if (
        isinstance(planned_tool, str)
        and planned_tool.strip()
        and isinstance(tool_name, str)
        and tool_name.strip() == planned_tool.strip()
    ):
        logger.debug(
            "reflection: planned tool '%s' succeeded — marking all requires _met (skipping LLM)",
            planned_tool,
        )
        for req in requires:
            if isinstance(req, str) and req.strip():
                accomplishments.append(f"{req.strip()}_met")
        return accomplishments, {}

    # 3. Different tool used — let LLM check if it semantically satisfies the requirements
    if agent_key:
        return _llm_check_requires_met(requires, execution_result, agent_key, tool_name)

    # Fallback: if execution succeeded and no agent_key, assume all met
    for req in requires:
        if isinstance(req, str) and req.strip():
            accomplishments.append(f"{req.strip()}_met")

    return accomplishments, {}


def _update_tool_call_history_with_accomplishments(
    state: AgentState,
    accomplishments: list[str],
) -> list[dict[str, Any]]:
    """Add accomplishments to the latest tool_call_history entry for current task.

    Finds the most recent entry matching current_task_id and adds accomplishments.
    If no entries exist, returns history unchanged.
    """
    tool_call_history = state.get("tool_call_history")
    if not isinstance(tool_call_history, list):
        return []

    if not accomplishments:
        return tool_call_history

    current_task_id = state.get("current_task")
    if not current_task_id:
        return tool_call_history

    # Find and update the latest entry for this task
    updated_history = []
    updated = False
    for i in range(len(tool_call_history) - 1, -1, -1):
        entry = tool_call_history[i]
        if not isinstance(entry, dict):
            continue

        if entry.get("task_id") == current_task_id and not updated:
            updated_entry = dict(entry)
            updated_entry["accomplishments"] = accomplishments
            updated_history.insert(0, updated_entry)
            updated = True
        else:
            updated_history.insert(0, entry)

    if not updated:
        return tool_call_history

    return updated_history




def _merge_token_stats(*stats_list: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "estimated_total": 0,
        "estimated_system": 0,
        "estimated_user": 0,
        "estimated_assistant": 0,
        "estimated_tool": 0,
        "estimated_other": 0,
        "message_count": 0,
        "llm_call_count": 0,
        "time_s": 0.0,
        "model_set": set(),
        "method": "fallback_estimated",
        "method_set": set(),
    }
    for stats in stats_list:
        if not isinstance(stats, dict) or not stats:
            continue
        merged["estimated_total"] += int(stats.get("estimated_total") or 0)
        merged["prompt_tokens"] += int(stats.get("prompt_tokens") or 0)
        merged["completion_tokens"] += int(stats.get("completion_tokens") or 0)
        merged["total_tokens"] += int(stats.get("total_tokens") or 0)
        merged["cached_tokens"] += int(stats.get("cached_tokens") or 0)
        merged["estimated_system"] += int(stats.get("estimated_system") or 0)
        merged["estimated_user"] += int(stats.get("estimated_user") or 0)
        merged["estimated_assistant"] += int(stats.get("estimated_assistant") or 0)
        merged["estimated_tool"] += int(stats.get("estimated_tool") or 0)
        merged["estimated_other"] += int(stats.get("estimated_other") or 0)
        merged["message_count"] += int(stats.get("message_count") or 0)
        merged["llm_call_count"] += int(stats.get("llm_call_count") or (1 if stats.get("estimated_total") else 0))
        merged["time_s"] += float(stats.get("time_s") or 0.0)
        model = str(stats.get("model") or "").strip()
        if model:
            merged["model_set"].add(model)
        method = str(stats.get("method") or "").strip()
        if method:
            merged["method_set"].add(method)
    models = sorted(merged["model_set"])
    merged["model"] = models[0] if len(models) == 1 else ("multiple" if models else "")
    merged.pop("model_set", None)
    methods = merged.pop("method_set", set())
    if methods == {"provider_reported"}:
        merged["method"] = "provider_reported"
    elif not methods:
        merged["method"] = "fallback_estimated"
    elif "provider_reported" in methods:
        merged["method"] = "mixed_provider_and_fallback"
    elif len(methods) == 1:
        merged["method"] = next(iter(methods))
    else:
        merged["method"] = "mixed_fallback"
    return merged


def _evidence_signature(entry: dict[str, Any]) -> str:
    """Dedup by tool_name + content hash — same tool returning same content is a duplicate."""
    tool_name = entry.get("tool_name") or ""
    content = json.dumps(entry.get("content"), sort_keys=True, ensure_ascii=True, default=str)
    return f"{tool_name}:{hash(content)}"


def _append_context_evidence(
    state: AgentState,
    new_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append new_entry to context_evidence, skipping duplicates by tool_name + content."""
    existing = state.get("context_evidence")
    history = list(existing) if isinstance(existing, list) else []
    new_sig = _evidence_signature(new_entry)
    if any(_evidence_signature(e) == new_sig for e in history if isinstance(e, dict)):
        return history
    history.append(new_entry)
    return history


def _has_substantial_text(value: Any) -> bool:
    if isinstance(value, str):
        return len(value.strip()) >= 80
    if isinstance(value, list):
        return any(_has_substantial_text(item) for item in value)
    if isinstance(value, dict):
        return any(
            _has_substantial_text(value.get(key))
            for key in ("content", "text", "body", "markdown")
        )
    return False


def _effective_evidence_role(role: str, tool_name: str | None, content: Any) -> str:
    """Only actual textual source material should be grounding evidence."""
    normalized_role = role if role in {"grounding", "alignment", "context"} else "context"
    if normalized_role != "grounding":
        return normalized_role

    normalized_tool = str(tool_name or "").casefold()
    if "list" in normalized_tool or "search" in normalized_tool:
        return "context"

    if not _has_substantial_text(content):
        return "context"

    return "grounding"


def reflection_node(state: AgentState) -> dict[str, Any]:
    """
    Evaluate the latest tool execution for the current task and write normalized state updates.

    High-level flow:
    - Read `execution_result`; if missing/failed, emit phase-aware `failure_type`
      (`missing_source` for source phase, `bad_artifact` for artifact phase).
    - Resolve current task from `task_plan` + `current_task`, then read `requires`
      and planned tool (`context.tool`).
    - Compute requirement accomplishments as `<requirement>_met` or
      `<requirement>_missing`:
      - Deterministic pass when planned tool succeeded.
      - Otherwise LLM semantic check when needed.
    - Attach accomplishments to the latest matching `tool_call_history` entry.
    - For source phase only: append deduplicated `context_evidence` when content is
      non-empty; otherwise set `failure_type="missing_source"`.
    - For artifact phase: skip evidence storage and return after accomplishment/metric updates.

    Returns:
    - A partial state delta containing some combination of:
      `failure_type`, `tool_call_history`, `context_evidence`, `metric_reflection`,
      and `metric_usage_events`.
    """
    execution_result = state.get("execution_result")
    metric_reflection: dict[str, Any] = {}
    current_phase = state.get("phase") or "source"

    if not isinstance(execution_result, dict):
        logger.debug("reflection: no execution_result in state")
        # No execution result = couldn't run tool at all
        failure_type = "bad_artifact" if current_phase == "artifact" else "missing_source"
        return {"failure_type": failure_type, "metric_reflection": metric_reflection, "metric_usage_events": []}

    if execution_result.get("success") is not True:
        logger.debug("reflection: tool execution failed: %s", execution_result.get("error"))
        # Phase-aware failure: artifact failures stay in artifact, source failures stay in source
        failure_type = "bad_artifact" if current_phase == "artifact" else "missing_source"
        return {"failure_type": failure_type, "metric_reflection": metric_reflection, "metric_usage_events": []}

    # Get task requirements to validate against
    task_plan = state.get("task_plan")
    current_task_id = state.get("current_task")
    task_requires: list[str] = []
    evidence_role = "context"
    planned_tool: str | None = None

    if isinstance(task_plan, list) and current_task_id:
        current_task = next((t for t in task_plan if t.get("id") == current_task_id), None)
        if isinstance(current_task, dict):
            task_requires = current_task.get("requires") or []
            raw_evidence_role = str(current_task.get("evidence_role") or "").strip().casefold()
            if raw_evidence_role in {"grounding", "alignment", "context"}:
                evidence_role = raw_evidence_role
            # Extract planned tool to enable deterministic accomplishment validation
            task_context = current_task.get("context")
            if isinstance(task_context, dict):
                tool_val = task_context.get("tool")
                if isinstance(tool_val, str) and tool_val.strip():
                    planned_tool = tool_val.strip()

    decision = state.get("active_tool_call") or state.get("decision")
    tool_name = decision.get("tool_name") if isinstance(decision, dict) else None
    content = execution_result.get("content")
    agent_key = state.get("agent_key") or "default"

    # Check which requires are satisfied by the execution result
    accomplishments: list[str] = []
    requires_token_stats: dict[str, Any] = {}
    if task_requires:
        accomplishments, requires_token_stats = _check_requires_satisfied(
            task_requires,
            execution_result,
            agent_key,
            tool_name=tool_name or "unknown",
            planned_tool=planned_tool,
        )
    update: dict[str, Any] = {}

    if accomplishments:
        update["tool_call_history"] = _update_tool_call_history_with_accomplishments(state, accomplishments)
        logger.debug("reflection: checked requires for task=%s: %s", current_task_id, accomplishments)

    # Skip evidence storage for artifact phase
    if state.get("phase") == "artifact":
        merged = _merge_token_stats(requires_token_stats)
        metric_reflection = build_metric(merged, float(merged.get("time_s") or 0.0), str(merged.get("model") or ""))
        update.update(build_metric_state_delta("reflection", "metric_reflection", metric_reflection))
        update["failure_type"] = None
        return update

    if not isinstance(decision, dict) or decision.get("action") != "use_tool":
        merged = _merge_token_stats(requires_token_stats)
        metric_reflection = build_metric(merged, float(merged.get("time_s") or 0.0), str(merged.get("model") or ""))
        update.update(build_metric_state_delta("reflection", "metric_reflection", metric_reflection))
        update["failure_type"] = None
        return update

    if content is None or (isinstance(content, str) and not content.strip()):
        logger.debug("reflection: tool returned empty content")
        merged = _merge_token_stats(requires_token_stats)
        metric_reflection = build_metric(merged, float(merged.get("time_s") or 0.0), str(merged.get("model") or ""))
        update.update(build_metric_state_delta("reflection", "metric_reflection", metric_reflection))
        update["failure_type"] = "missing_source"
        return update

    # Content is already cleaned by execution_node before being stored in state
    if content is None or (isinstance(content, str) and not str(content).strip()):
        logger.debug("reflection: content empty")
        merged = _merge_token_stats(requires_token_stats)
        metric_reflection = build_metric(merged, float(merged.get("time_s") or 0.0), str(merged.get("model") or ""))
        update.update(build_metric_state_delta("reflection", "metric_reflection", metric_reflection))
        update["failure_type"] = "missing_source"
        return update

    task_id = state.get("current_task") or "unknown"

    new_entry = {
        "task_id": task_id,
        "evidence_role": _effective_evidence_role(evidence_role, tool_name, content),
        "tool_name": tool_name if isinstance(tool_name, str) else "",
        "content": content,
    }

    update["context_evidence"] = _append_context_evidence(state, new_entry)
    merged = _merge_token_stats(requires_token_stats)
    update.update(
        build_metric_state_delta(
            "reflection",
            "metric_reflection",
            build_metric(merged, float(merged.get("time_s") or 0.0), str(merged.get("model") or "")),
        )
    )
    update["failure_type"] = None

    logger.debug("reflection: stored evidence for tool=%s", tool_name)
    return update
