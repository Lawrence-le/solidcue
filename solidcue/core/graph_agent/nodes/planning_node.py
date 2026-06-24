import json
import logging
import re
from typing import Any

from solidcue.agent_configs.loader import load_agent, load_agent_skill, load_agent_tools, get_task_plan_path
from solidcue.providers.provider_resolver import get_provider_for_role
from solidcue.core.graph_agent.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_async_stream_generate
from solidcue.core.graph_agent.prompts.planning_prompt import build_planning_messages

logger = logging.getLogger(__name__)
_VAGUE_REQUIRES = {"data", "details", "information", "context", "output", "done"}
_ACTION_PREFIXES = ("execute ", "download ", "list ", "run ", "call ", "use ")


def _format_chat_history(chat_history: list[dict[str, Any]] | None, *, limit: int = 8) -> str:
    if not isinstance(chat_history, list) or not chat_history:
        return "None"

    lines: list[str] = []
    for entry in chat_history[-limit:]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip()
        content = str(entry.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"

"""
Planning Node - Function Overview
---------------------------------

_extract_json_object:
Safely parse JSON object from planner LLM output, even when wrapped in prose.

_llm_plan:
Request raw task list from planner model.

_fallback_task_plan:
Create deterministic fallback tasks when model planning fails.

_guardrail_renumber_task_ids:
Normalize task IDs into sequential `task_N`.

_guardrail_assign_item_keys:
Ensure every task has `context.item_key` for item-scoped downstream execution.

_slug:
Convert free-form text into snake_case-safe token.

_normalize_requires_for_task:
Normalize/validate `requires` labels with safe defaults by task type.

_guardrail_normalize_task_shape:
Coerce planner output into canonical runtime task shape.

_apply_planning_guardrails:
Pipeline wrapper: normalize shape -> renumber IDs -> assign item keys.

_build_raw_task_plan:
Select planning source path (empty-input default vs LLM/fallback) and flag
whether the result is cacheable (LLM-derived only).

planning_node:
Main entrypoint. Phases:
1) Reuse cached guardrailed template if present
2) Otherwise build raw plan, apply guardrails, and cache the guardrailed result
3) Return normalized plan + metrics
"""

# ---------------------------------------------------------------------------
# Section: parser + LLM planning
# ---------------------------------------------------------------------------


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


async def _llm_plan(state: AgentState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            available_tools=list(getattr(agent, "tools", []) or []),
            source_paths=state.get("source_paths") or [],
            output_paths=state.get("output_paths") or [],
            source_filenames=state.get("source_filenames") or [],
            output_filenames=state.get("output_filenames") or [],
            chat_history=state.get("chat_history") or [],
        )
        response_text, metric_stats = await timed_async_stream_generate(provider, messages, node_name="planning")
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


# ---------------------------------------------------------------------------
# Section: task-id + item-key helpers
# ---------------------------------------------------------------------------

def _guardrail_renumber_task_ids(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure task IDs are strictly sequential using task_N format."""
    normalized: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        if isinstance(task, dict):
            updated = {**task, "id": f"task_{idx}"}
            normalized.append(updated)
    return normalized


def _ordinal_index_from_text(text: str) -> int | None:
    lowered = str(text or "").casefold()
    mapping = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "1st": 1,
        "2nd": 2,
        "3rd": 3,
        "4th": 4,
        "5th": 5,
    }
    for token, idx in mapping.items():
        if token in lowered:
            return idx
    return None


def _metadata_item_map(target_artifacts_source: list[dict[str, Any]] | None) -> tuple[dict[int, str], dict[str, str]]:
    by_index: dict[int, str] = {}
    by_url: dict[str, str] = {}
    if not isinstance(target_artifacts_source, list):
        return by_index, by_url
    for item in target_artifacts_source:
        if not isinstance(item, dict):
            continue
        key = str(item.get("item_key") or "").strip()
        if not key:
            continue
        index = item.get("index")
        if isinstance(index, int) and index > 0:
            by_index[index] = key
        source_ref = str(item.get("source_ref") or "").strip()
        if source_ref:
            by_url[source_ref] = key
    return by_index, by_url


def _guardrail_assign_item_keys(
    tasks: list[dict[str, Any]],
    target_artifacts_source: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Ensure each task has a stable `context.item_key`.

    Priority:
    1) Keep explicit context.item_key if present.
    2) Resolve from discovery map via context.source_item_index (input-only).
    3) Resolve from discovery map via URL-like context fields.
    4) Resolve from discovery map via ordinal wording in task description.
    5) Derive from URL-like context fields (stable hash key).

    6) Inherit the most recent URL-derived/explicit key for adjacent non-URL tasks.
    7) Fallback to deterministic sequence key.

    After the key is resolved, all user-input source values (URL-like fields and
    any value matching a known source_ref) are stripped from context so the plan
    stays request-agnostic and reusable. Downstream nodes bind the concrete
    source value from target_artifacts_source via item_key.
    """
    url_fields = ("source_ref", "posting_url", "url", "jd_url", "job_url")
    url_to_item: dict[str, str] = {}
    metadata_index_map, metadata_url_map = _metadata_item_map(target_artifacts_source)
    known_source_refs = {
        str(item.get("source_ref") or "").strip()
        for item in (target_artifacts_source or [])
        if isinstance(item, dict) and str(item.get("source_ref") or "").strip()
    }
    discovered_keys: list[str] = []
    current_item_key: str | None = None
    fallback_counter = 0
    normalized: list[dict[str, Any]] = []

    for task in tasks:
        if not isinstance(task, dict):
            continue
        context = task.get("context") if isinstance(task.get("context"), dict) else {}
        task_context = dict(context)
        resolved_index: int | None = None

        explicit_key = str(task_context.get("item_key") or "").strip()
        item_key = _slug(explicit_key) if explicit_key else ""

        index_value = task_context.get("source_item_index")
        if isinstance(index_value, int):
            resolved_index = index_value
        elif isinstance(index_value, str) and index_value.strip().isdigit():
            resolved_index = int(index_value.strip())

        if not item_key and isinstance(resolved_index, int):
            item_key = metadata_index_map.get(resolved_index, "")

        if not item_key:
            matched_url = ""
            for field in url_fields:
                candidate = task_context.get(field)
                if isinstance(candidate, str) and candidate.strip():
                    matched_url = candidate.strip()
                    break
            if matched_url:
                mapped_key = (
                    metadata_url_map.get(matched_url)
                    or url_to_item.get(matched_url)
                    # Positional fallback (a slot, not a source identity) so the
                    # plan stays reusable across requests with different sources.
                    or f"item_{len(url_to_item) + 1}"
                )
                item_key = mapped_key
                url_to_item[matched_url] = mapped_key
                if metadata_index_map and not isinstance(resolved_index, int):
                    for idx, key in metadata_index_map.items():
                        if key == mapped_key:
                            resolved_index = idx
                            break

        if not item_key:
            ordinal_index = _ordinal_index_from_text(task.get("description"))
            if isinstance(ordinal_index, int):
                resolved_index = ordinal_index
                item_key = metadata_index_map.get(ordinal_index, "")

        if not item_key and current_item_key:
            item_key = current_item_key

        if not item_key and len(discovered_keys) == 1:
            item_key = discovered_keys[0]

        if not item_key:
            fallback_counter += 1
            item_key = f"item_{fallback_counter}"

        if item_key not in discovered_keys:
            discovered_keys.append(item_key)
        current_item_key = item_key

        task_context["item_key"] = item_key
        # Keep source_item_index as an internal planning hint only.
        # Runtime task context should bind by item_key only.
        task_context.pop("source_item_index", None)

        # Strip user-input source values so the plan stays request-agnostic.
        # The source binding is carried solely by item_key; the concrete value
        # (URL/path) is resolved downstream from target_artifacts_source. Leaving
        # it here would pollute a reused plan when a future request has a
        # different source.
        for field in url_fields:
            task_context.pop(field, None)
        for key in list(task_context.keys()):
            if key == "item_key":
                continue
            value = task_context.get(key)
            if isinstance(value, str) and value.strip() in known_source_refs:
                task_context.pop(key, None)

        normalized.append({**task, "context": task_context})

    return normalized


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).casefold()).strip("_")
    return value or "item"


# ---------------------------------------------------------------------------
# Section: task-shape normalization helpers
# ---------------------------------------------------------------------------

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
                "context": context,
                "status": str(task.get("status") or "pending"),
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Section: task plan cache helpers
# ---------------------------------------------------------------------------


def _load_task_plan_cache(agent_key: str) -> list[dict[str, Any]] | None:
    path = get_task_plan_path(agent_key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _save_task_plan_cache(agent_key: str, tasks: list[dict[str, Any]]) -> None:
    path = get_task_plan_path(agent_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    except Exception:
        pass


def _plan_is_cacheable(agent_key: str) -> bool:
    """Whether this agent's task plan may be cached and reused.

    Only `static` (deterministic-pipeline) agents cache their plan. `dynamic`
    agents re-plan every turn because the plan shape — not just its inputs —
    varies per request, so a cached plan would be replayed incorrectly.
    Defaults to non-cacheable on any load failure (the safe direction).
    """
    if not agent_key:
        return False
    try:
        return load_agent(agent_key).planning.mode == "static"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Section: planning pipeline helpers
# ---------------------------------------------------------------------------

def _apply_planning_guardrails(
    tasks: list[dict[str, Any]],
    target_artifacts_source: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize planner output into canonical runtime shape.

    Pipeline (order matters):
    1) Normalize task shape/types/context/requires
    2) Renumber task ids deterministically
    3) Inject per-item `context.item_key` (using discovery item map when available)
    """
    normalized = _guardrail_normalize_task_shape(tasks)
    renumbered = _guardrail_renumber_task_ids(normalized)
    with_item_keys = _guardrail_assign_item_keys(renumbered, target_artifacts_source=target_artifacts_source)
    return with_item_keys


async def _build_raw_task_plan(state: AgentState) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    """Build the raw (pre-guardrail) task plan, metrics, and a cacheable flag.

    The cacheable flag is True only for LLM-derived plans. The empty-input
    default and the deterministic fallback are not cached, since they are cheap
    to recompute and not representative of the agent's real workflow.
    """
    user_input = state.get("user_input", "")
    if not user_input.strip():
        return (
            [
                {
                    "id": "task_1",
                    "type": "synthesis",
                    "description": "Respond to user",
                    "requires": ["final_response_draft_text"],
                    "status": "pending",
                }
            ],
            {},
            False,
        )

    task_plan, metric_planning = await _llm_plan(state)
    if not task_plan:
        return _fallback_task_plan(state), metric_planning, False
    return task_plan, metric_planning, True


async def planning_node(state: AgentState) -> dict[str, Any]:
    """Generate a structured task plan for downstream execution."""
    agent_key = state.get("agent_key") or ""

    # Phase 1: reuse the cached, already-guardrailed template if present. The
    # cache stores the source-agnostic plan (URLs stripped, positional item_keys),
    # so it is reusable across requests with different sources and needs no
    # re-normalization here.
    # The cache is only consulted for `static` agents. `dynamic` agents re-plan
    # every turn, so a stale request-specific plan is never read back or written.
    plan_cacheable = _plan_is_cacheable(agent_key)
    cached = _load_task_plan_cache(agent_key) if plan_cacheable else None
    if cached is not None:
        task_plan = cached
        metric_planning: dict[str, Any] = {}
    else:
        # Phase 2: build raw plan, normalize it, and cache the guardrailed result
        # (never the raw LLM output, which still carries request-specific values).
        raw_task_plan, metric_planning, cacheable = await _build_raw_task_plan(state)
        task_plan = _apply_planning_guardrails(
            raw_task_plan,
            target_artifacts_source=state.get("target_artifacts_source") or [],
        )
        if plan_cacheable and cacheable:
            _save_task_plan_cache(agent_key, task_plan)

    # Phase 3: finalize planning state for downstream nodes.
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
