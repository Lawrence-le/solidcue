import json
import logging
import hashlib
from typing import Any, cast

logger = logging.getLogger(__name__)

from solidcue.agents.configs.loader import load_agent, load_agent_persona, load_agent_skill
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate
from solidcue.prompts.synthesis_prompt import build_synthesis_messages

# --- RESILIENT DATA EXTRACTION UTILITIES ---

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

def _build_source_from_handoff(state: AgentState) -> str:
    """Extract readable source content from the handoff.

    Returns text entries keyed by their requires label so the LLM
    knows which source each block came from.
    """
    handoff = state.get("handoff")
    if not isinstance(handoff, dict) or not handoff:
        return ""

    sections: list[str] = []
    for key, value in handoff.items():
        if isinstance(value, str) and value.strip():
            sections.append(f"=== {key} ===\n{value}")
        elif isinstance(value, dict):
            text = value.get("text") or value.get("content") or value.get("body")
            if isinstance(text, str) and text.strip():
                sections.append(f"=== {key} ===\n{text}")
    return "\n\n".join(sections)


def _build_deduplicated_context(state: AgentState) -> str:
    """
    Extracts source material, preferring handoff (full content) over
    context_evidence (may be truncated).
    """
    handoff_material = _build_source_from_handoff(state)
    if handoff_material:
        return handoff_material

    evidence = state.get("context_evidence") or []
    if not evidence and state.get("execution_result"):
        evidence = [state.get("execution_result")]

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


# --- CORE NODE ---

def _write_draft_to_handoff(state: AgentState, draft: str) -> dict[str, Any]:
    """Store synthesis_draft in the handoff so artifact tasks can consume it."""
    handoff = dict(state.get("handoff") or {})
    handoff["synthesis_draft"] = {"content": draft}
    return handoff


def synthesis_node(state: AgentState) -> dict[str, Any]:
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

    # 2. Gather and Clean Source Material
    source_material = _build_deduplicated_context(state)
    
    # Fallback to user input if no evidence was gathered
    if not source_material:
        source_material = str(state.get("user_input") or "No data gathered.")

    # 4. LLM Synthesis
    agent_key = state.get("agent_key")
    if not agent_key:
        return _create_response(source_material)

    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "writer")
        
        # Load persona/skills with fail-safes
        persona = ""
        skill = ""
        try:
            persona = load_agent_persona(agent_key)
            skill = load_agent_skill(agent_key)
        except Exception as e:
            logger.warning(f"Metadata load failed for {agent_key}: {e}")

        task_description = ""
        skill_section = ""
        task_plan = state.get("task_plan")
        current_task_id = state.get("current_task")
        if isinstance(task_plan, list) and current_task_id:
            current_task = next((t for t in task_plan if isinstance(t, dict) and t.get("id") == current_task_id), None)
            if current_task:
                task_description = str(current_task.get("description") or "")
                ctx = current_task.get("context")
                if isinstance(ctx, dict):
                    skill_section = str(ctx.get("skill_section") or "")

        skill_for_synthesis = _extract_skill_section(skill, skill_section) if skill_section else skill

        messages = build_synthesis_messages(
            user_query=str(state.get("user_input", "")),
            raw_data=source_material,
            metadata=state.get("metadata"),
            retry_reason=state.get("retry_reason"),
            persona_text=persona,
            skill_text=skill_for_synthesis,
            task_description=task_description,
        )
        polished, metric_synthesis = timed_generate(provider, messages)
        polished_text = str(polished or "").strip()

        # Final Safety Check: If LLM failed or produced nonsense, use the cleaned material
        final_draft = polished_text if (polished_text and not (polished_text.startswith("{") and "}" in polished_text)) else source_material
        return _create_response(final_draft, metric_synthesis)

    except Exception as e:
        logger.exception(f"Synthesis LLM failed: {e}")
        return _create_response(source_material)

