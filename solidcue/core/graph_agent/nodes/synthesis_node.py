import json
import logging
import hashlib
from typing import Any

logger = logging.getLogger(__name__)

from solidcue.agent_configs.loader import load_agent, load_agent_persona, load_agent_skill
from solidcue.providers.provider_resolver import get_provider_for_role
from solidcue.core.graph_agent.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_async_stream_generate
from solidcue.core.graph_agent.prompts.synthesis_prompt import build_synthesis_messages

"""
Synthesis Node - Function Overview
----------------------------------

_is_noise_key:
Filters low-value metadata keys from tool payloads.

_extract_meaningful_content:
Recursively removes noise and flattens simple dict wrappers.

_get_current_task:
Resolves current task from state.task_plan + state.current_task.

_get_item_key_from_task:
Reads context.item_key from current task (if present).

_handoff_for_item:
Builds an item-scoped handoff view:
- item-bound entries from `::<item_key>`
- shared entries from `global::`

_build_source_from_handoff:
Converts handoff data into readable synthesis input blocks.

_build_deduplicated_context:
Builds fallback context from execution_result and deduplicates.

_build_artifact_delivery_message:
Formats deterministic artifact confirmation text (utility helper).

_extract_skill_section:
Extracts one section from SKILL.md for focused synthesis guidance.

_write_draft_to_handoff:
Stores synthesis draft into handoff using scoped/global keying.

synthesis_node:
Main orchestration entrypoint. Phases:
1) Build source material
2) Resolve agent/provider + persona/skills
3) Generate draft
4) Persist draft + metrics + handoff
"""

# ---------------------------------------------------------------------------
# Section: resilient data extraction helpers
# ---------------------------------------------------------------------------

def _is_noise_key(key: str) -> bool:
    """Identify technical metadata keys that distract the LLM."""
    noise = {
        "id", "fileid", "documentid", "status", "ok", "browser", 
        "mimetype", "sourcemimetype", "webviewlink", "alternatelink",
        "nextpagetoken", "resolvedfolderid", "webcontentlink", "success"
    }
    return str(key).lower() in noise

def _extract_meaningful_content(data: Any) -> Any:
    """
    Recursively strips technical noise and flattens single-key dictionaries.
    This handles unpredictable tool shapes.
    """
    if isinstance(data, list):
        return [_extract_meaningful_content(item) for item in data if item]
    
    if isinstance(data, dict):
        # Filter out metadata/noise
        cleaned = {
            k: _extract_meaningful_content(v) 
            for k, v in data.items() if not _is_noise_key(k)
        }
        
        # If the dictionary is empty after cleaning, return None
        if not cleaned:
            return None
        # If only one key remains (e.g., {"text": "..."}), flatten it
        if len(cleaned) == 1:
            return list(cleaned.values())[0]
        return cleaned
    
    return data


# ---------------------------------------------------------------------------
# Section: task / item scope helpers
# ---------------------------------------------------------------------------

def _get_current_task(state: AgentState) -> dict[str, Any] | None:
    task_plan = state.get("task_plan")
    current_task_id = state.get("current_task")
    if not isinstance(task_plan, list) or not current_task_id:
        return None
    return next((t for t in task_plan if isinstance(t, dict) and t.get("id") == current_task_id), None)


def _get_item_key_from_task(task: dict[str, Any] | None) -> str | None:
    if not isinstance(task, dict):
        return None
    context = task.get("context")
    if not isinstance(context, dict):
        return None
    item_key = context.get("item_key")
    if not isinstance(item_key, str):
        return None
    cleaned = item_key.strip()
    return cleaned or None


def _handoff_for_item(handoff: dict[str, Any], item_key: str | None) -> dict[str, Any]:
    """Build scoped handoff view for synthesis.

    Returns only:
    - item-scoped entries for this item key (suffix stripped)
    - shared global entries (prefix stripped)
    """
    scoped: dict[str, Any] = {}
    if item_key:
        suffix = f"::{item_key}"
        for key, value in handoff.items():
            if isinstance(key, str) and key.endswith(suffix):
                scoped[key[: -len(suffix)]] = value

    for key, value in handoff.items():
        if isinstance(key, str) and key.startswith("global::"):
            scoped[key[len("global::"):]] = value

    return scoped


# ---------------------------------------------------------------------------
# Section: source material builders
# ---------------------------------------------------------------------------

# Prefix under which per-item identity labels are stored in the handoff.
_ITEM_LABEL_KEY = "item_label"


def _stringify_handoff_value(value: Any) -> str:
    """Render a handoff value as source text.

    Prefers a text-shaped field (`text`/`content`/`body`); otherwise serializes
    the whole value so structured (non-text) payloads are surfaced instead of
    dropped. Branches on value SHAPE only — no domain-specific field names.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or value.get("body")
        if isinstance(text, str) and text.strip():
            return text.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, list):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return ""


def _is_aggregate_task(task: dict[str, Any] | None) -> bool:
    """Whether this synthesis task aggregates across all items.

    Decided upstream by the planner via a neutral `context.scope` flag; core
    never inspects the user request to guess.
    """
    if not isinstance(task, dict):
        return False
    context = task.get("context")
    scope = context.get("scope") if isinstance(context, dict) else None
    return isinstance(scope, str) and scope.strip().casefold() in {"all", "aggregate"}


def _build_aggregate_source(handoff: dict[str, Any]) -> str:
    """Render every item's handoff entries, grouped and labeled per item.

    Generic: iterates item_key slots, uses the stored `item_label` (if any) as
    the heading, falling back to the slot key. No domain knowledge.
    """
    groups: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}
    shared: dict[str, Any] = {}

    for key, value in handoff.items():
        if not isinstance(key, str) or "::" not in key:
            continue
        left, _, right = key.partition("::")
        if left == "global":
            shared[right] = value
        elif left == _ITEM_LABEL_KEY:
            labels[right] = value if isinstance(value, str) else str(value)
        else:
            groups.setdefault(right, {})[left] = value

    sections: list[str] = []
    for item_key in sorted(groups):
        label = labels.get(item_key) or item_key
        lines: list[str] = []
        for base, value in groups[item_key].items():
            block = _stringify_handoff_value(value)
            if block:
                lines.append(f"{base}: {block}")
        if lines:
            sections.append(f"=== {label} ===\n" + "\n".join(lines))

    for base, value in shared.items():
        block = _stringify_handoff_value(value)
        if block:
            sections.append(f"=== {base} (shared) ===\n{block}")

    return "\n\n".join(sections)


def _build_source_from_handoff(state: AgentState) -> str:
    """Extract readable source content from the handoff.

    Aggregate synthesis tasks read every item (labeled per item); otherwise the
    view is scoped to the task's single item_key. Structured values are
    serialized, not dropped.
    """
    handoff = state.get("handoff")
    if not isinstance(handoff, dict) or not handoff:
        return ""

    task = _get_current_task(state)
    if _is_aggregate_task(task):
        return _build_aggregate_source(handoff)

    item_key = _get_item_key_from_task(task)
    handoff_view = _handoff_for_item(handoff, item_key)

    sections: list[str] = []
    for key, value in handoff_view.items():
        block = _stringify_handoff_value(value)
        if block:
            sections.append(f"=== {key} ===\n{block}")
    return "\n\n".join(sections)


def _build_deduplicated_context(state: AgentState) -> str:
    """
    Extracts source material, preferring handoff (full content) over
    execution_result fallback.
    """
    handoff_material = _build_source_from_handoff(state)
    if handoff_material:
        return handoff_material

    evidence = [state.get("execution_result")] if state.get("execution_result") else []

    processed_items = []
    seen_hashes = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        raw_payload = item.get("content") or item
        meaningful_data = _extract_meaningful_content(raw_payload)

        if meaningful_data is None:
            continue

        content_str = json.dumps(meaningful_data, sort_keys=True, ensure_ascii=False)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()

        if content_hash not in seen_hashes:
            processed_items.append(meaningful_data)
            seen_hashes.add(content_hash)

    if not processed_items:
        return ""

    return json.dumps(processed_items, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Section: output/prompt utilities
# ---------------------------------------------------------------------------

def _build_artifact_delivery_message(content: Any) -> str | None:
    """Deterministic confirmation for document creation."""
    if not isinstance(content, dict):
        return None

    url = content.get("url") or content.get("webViewLink") or content.get("alternateLink")
    title = content.get("title") or content.get("name") or "Document"
    
    if url:
        return f"Done. Your document has been created: **[{title}]({url})**"
    return None

def _extract_skill_section(skill_text: str, section_name: str) -> str:
    """Extract a single top-level section from SKILL.md by its heading label.

    Looks for `# [SECTION_NAME]` and returns everything from that heading
    up to the next `# [` heading or end of file.
    """
    if not skill_text or not section_name:
        return skill_text or ""

    import re
    pattern = rf"(^# \[{re.escape(section_name)}\].*?)(?=^# \[|\Z)"
    match = re.search(pattern, skill_text, flags=re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else skill_text


# ---------------------------------------------------------------------------
# Section: handoff write helper
# ---------------------------------------------------------------------------

def _write_draft_to_handoff(state: AgentState, draft: str) -> dict[str, Any]:
    """Store synthesis draft in handoff using scoped/global keying."""
    handoff = dict(state.get("handoff") or {})
    task = _get_current_task(state)
    item_key = _get_item_key_from_task(task)
    if item_key:
        handoff[f"synthesis_draft::{item_key}"] = {"content": draft}
    else:
        handoff["global::synthesis_draft"] = {"content": draft}
    return handoff


# ---------------------------------------------------------------------------
# Section: core node
# ---------------------------------------------------------------------------

async def synthesis_node(state: AgentState) -> dict[str, Any]:
    """
    Produces synthesis_draft from collected evidence.
    Task completion is handled by the router after validation passes.
    """
    def _create_response(draft: str, metric_synthesis: dict[str, Any] | None = None) -> dict[str, Any]:
        stats = metric_synthesis or {}
        return {
            "synthesis_draft": draft,
            **build_metric_state_delta("synthesis", "metric_synthesis", stats),
            "handoff": _write_draft_to_handoff(state, draft),
        }

    # Phase 1: Gather and clean source material.
    source_material = _build_deduplicated_context(state)
    
    # Fallback: if no evidence was gathered, use user input as minimal source.
    if not source_material:
        source_material = str(state.get("user_input") or "No data gathered.")

    # Phase 2: Resolve agent/provider context.
    agent_key = state.get("agent_key")
    if not agent_key:
        return _create_response(source_material)

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "writer")
        
        # Phase 3: Load persona/skill guidance (best effort).
        persona = ""
        skill = ""
        try:
            persona = load_agent_persona(agent_key)
            skill = load_agent_skill(agent_key)
        except Exception as e:
            logger.warning(f"Metadata load failed for {agent_key}: {e}")

        task_description = ""
        skill_section = ""
        current_task = _get_current_task(state)
        if current_task:
            task_description = str(current_task.get("description") or "")
            ctx = current_task.get("context")
            if isinstance(ctx, dict):
                skill_section = str(ctx.get("skill_section") or "")

        skill_for_synthesis = _extract_skill_section(skill, skill_section) if skill_section else skill

        # Phase 4: Build synthesis prompt and generate draft.
        messages = build_synthesis_messages(
            user_query=str(state.get("user_input", "")),
            raw_data=source_material,
            metadata=state.get("metadata"),
            retry_reason=state.get("retry_reason"),
            persona_text=persona,
            skill_text=skill_for_synthesis,
            task_description=task_description,
        )
        polished, metric_synthesis = await timed_async_stream_generate(provider, messages, node_name="synthesis")
        polished_text = str(polished or "").strip()

        # Safety fallback: if generation looks invalid, preserve source material.
        final_draft = polished_text if (polished_text and not (polished_text.startswith("{") and "}" in polished_text)) else source_material
        return _create_response(final_draft, metric_synthesis)

    except Exception as e:
        logger.exception(f"Synthesis LLM failed: {e}")
        return _create_response(source_material)
