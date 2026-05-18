"""Task planning node: decomposes user requests into concrete steps.

Phase 3 enhancement: Takes a user query and generates a structured task plan
to guide multi-step workflows. Each task has a type, description, and status.
"""

import json
import logging
import re
from typing import Any

from solidcue.agents.configs.loader import load_agent, load_agent_skill, load_agent_tools
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate
from solidcue.prompts.planning_prompt import build_planning_messages

logger = logging.getLogger(__name__)
_VAGUE_REQUIRES = {"data", "details", "information", "context", "output", "done"}
_ACTION_PREFIXES = ("execute ", "download ", "list ", "run ", "call ", "use ")
_EVIDENCE_ROLES = {"grounding", "alignment", "context"}
_GROUNDING_HINTS = (
    "candidate",
    "cv",
    "linkedin_profile",
    "master_resume",
    "profile",
    "resume_master",
    "work_history",
)
_ALIGNMENT_HINTS = (
    "audience",
    "criteria",
    "jd",
    "job",
    "job_description",
    "posting",
    "requirements",
    "role",
    "rubric",
    "target",
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _llm_plan(state: AgentState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Call LLM to generate a task plan from user input.

    Returns a list of tasks with type, description, requires, and status.
    """
    user_input = state.get("user_input", "")

    try:
        agent_key = state.get("agent_key")
        if not isinstance(agent_key, str) or not agent_key:
            return [], {}

        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "brain")
        messages = build_planning_messages(
            user_input=user_input,
            skill_guidance=load_agent_skill(agent_key),
            tools_guidance=load_agent_tools(agent_key),
            metadata=state.get("metadata") if isinstance(state.get("metadata"), dict) else {},
        )
        response_text, metric_stats = timed_generate(provider, messages)
        logger.debug("planning LLM raw response (type=%s): %s", type(response_text).__name__, repr(response_text)[:500])

        parsed = _extract_json_object(str(response_text or ""))
        if not isinstance(parsed, dict):
            logger.warning("planning LLM response did not parse as JSON, falling back")
            return [], metric_stats

        tasks = parsed.get("tasks", [])
        if isinstance(tasks, list):
            return [task for task in tasks if isinstance(task, dict)], metric_stats
    except Exception as e:
        logger.warning(f"LLM task planning failed, using fallback: {e}")

    return [], {}


def _fallback_task_plan(state: AgentState) -> list[dict[str, Any]]:
    """Generate a simple fallback task plan when LLM fails."""
    user_input = state.get("user_input", "").lower()

    # Detect if user is asking for artifact generation (resume, document, etc.)
    artifact_keywords = {"resume", "document", "generate", "create", "write", "build", "make"}
    needs_artifact = any(kw in user_input for kw in artifact_keywords)

    tasks = []

    if needs_artifact:
        tasks.append({
            "id": "task_1",
            "type": "source_gathering",
            "description": "Gather information from provided sources",
            "requires": ["source_context_text_content"],
            "status": "pending"
        })
        tasks.append({
            "id": "task_2",
            "type": "artifact_generation",
            "description": "Generate the requested artifact",
            "requires": ["artifact_delivery_confirmation"],
            "status": "pending"
        })

    tasks.append({
        "id": f"task_{len(tasks) + 1}",
        "type": "synthesis",
        "description": "Synthesize final response",
        "requires": ["final_response_draft_text"],
        "status": "pending"
    })

    return tasks


def _guardrail_renumber_task_ids(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure task IDs are strictly sequential using task_N format."""
    normalized: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        if isinstance(task, dict):
            updated = {**task, "id": f"task_{idx}"}
            normalized.append(updated)
    return normalized


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).casefold()).strip("_")
    return value or "item"


def _normalize_requires_for_task(task_type: str, requires: Any) -> list[str]:
    items = [str(x).strip() for x in requires] if isinstance(requires, list) else []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item:
            continue
        lowered = item.casefold().strip()
        if lowered in _VAGUE_REQUIRES:
            continue
        if lowered.startswith(_ACTION_PREFIXES):
            # Convert action phrasing into outcome-style key
            lowered = f"{_slug(lowered)}_result"
        key = _slug(lowered)
        if key and key not in seen:
            seen.add(key)
            normalized.append(key)

    if normalized:
        return normalized

    if task_type == "source_gathering":
        return ["source_text_content_collected"]
    if task_type in {"synthesis", "review"}:
        return ["synthesis_draft_content_ready"]
    if task_type == "artifact_generation":
        return ["artifact_delivery_confirmation"]
    return ["task_result_ready"]


def _infer_evidence_role(task_type: str, task: dict[str, Any], requires: list[str]) -> str:
    if task_type != "source_gathering":
        return "context"

    explicit = str(task.get("evidence_role") or "").strip().casefold()
    if explicit in _EVIDENCE_ROLES:
        return explicit

    haystack_parts = [
        str(task.get("description") or ""),
        " ".join(requires),
    ]
    raw_context = task.get("context")
    if isinstance(raw_context, dict):
        haystack_parts.extend(str(value) for value in raw_context.values())
    haystack = " ".join(haystack_parts).casefold()

    if any(hint in haystack for hint in _GROUNDING_HINTS):
        return "grounding"
    if any(hint in haystack for hint in _ALIGNMENT_HINTS):
        return "alignment"
    return "context"


def _guardrail_normalize_task_shape(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_type = str(task.get("type") or "synthesis").strip().casefold()
        if task_type not in {"source_gathering", "artifact_generation", "synthesis", "review"}:
            task_type = "synthesis"
        raw_context = task.get("context")
        context: dict[str, Any] = {}
        if isinstance(raw_context, dict):
            for k, v in raw_context.items():
                key = str(k).strip()
                if not key:
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    context[key] = v
                else:
                    context[key] = str(v)
        requires = _normalize_requires_for_task(task_type, task.get("requires"))
        normalized.append(
            {
                "id": str(task.get("id") or ""),
                "type": task_type,
                "description": str(task.get("description") or f"Execute {task_type} task"),
                "requires": requires,
                "evidence_role": _infer_evidence_role(task_type, task, requires),
                "context": context,
                "status": str(task.get("status") or "pending"),
            }
        )
    return normalized


def _conversational_response(state: AgentState) -> dict[str, Any]:
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {"final_response": "", "router_next": "final_output"}

    user_input = state.get("user_input", "")
    agent = load_agent(agent_key)
    skill = load_agent_skill(agent_key)
    tools = load_agent_tools(agent_key)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are {agent.name or 'an AI agent'}. {agent.description or ''}\n\n"
                f"# Your Skills\n{skill or 'General assistant.'}\n\n"
                f"# Available Tools\n{tools or 'No specific tools.'}\n\n"
                "Answer the user's question directly and helpfully using the information above. "
                "Keep it concise."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    provider = get_provider_for_role(agent, "lite")
    response_text, metric_stats = timed_generate(provider, messages)

    return {
        "phase": "conversational",
        "router_next": "final_output",
        "final_response": str(response_text or ""),
        "task_plan": [],
        "current_task": "",
        **build_metric_state_delta("planning", "metric_planning", metric_stats),
        "messages": [{"role": "system", "content": "Conversational question answered"}],
    }


def planning_node(state: AgentState) -> dict[str, Any]:
    """Generate a task plan from user input, or answer conversational questions.

    If phase is 'conversational' (routed from classifier), answers the user's
    question using skill and tools context. Otherwise decomposes the request
    into a structured task plan.
    """
    if state.get("phase") == "conversational":
        return _conversational_response(state)

    user_input = state.get("user_input", "")

    if not user_input.strip():
        task_plan = [{
            "id": "task_1",
            "type": "synthesis",
            "description": "Respond to user",
            "requires": ["final_response_draft_text"],
            "status": "pending"
        }]
    else:
        task_plan, metric_planning = _llm_plan(state)
        if not task_plan:
            task_plan = _fallback_task_plan(state)
    if not user_input.strip():
        metric_planning = {}

    task_plan = _guardrail_normalize_task_shape(task_plan)
    task_plan = _guardrail_renumber_task_ids(task_plan)

    first_task_id = task_plan[0]["id"] if task_plan else "task_1"
    return {
        "task_plan": task_plan,
        "current_task": first_task_id,
        **build_metric_state_delta("planning", "metric_planning", metric_planning),
        "messages": [
            {
                "role": "system",
                "content": f"Generated task plan with {len(task_plan)} tasks"
            }
        ]
    }
