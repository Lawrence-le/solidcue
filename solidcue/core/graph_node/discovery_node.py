from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from solidcue.agents.configs.loader import load_agent, load_agent_persona, load_agent_skill, load_agent_tools
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate
from solidcue.services.chat_history_service import load_chat_history

"""
Discovery Node - Function Overview
----------------------------------

_extract_json_object:
Parse discovery/model JSON payloads robustly.

_extract_paths_with_llm:
Extract path/filename hints from SKILL/TOOLS guidance.

discovery_node:
Main entrypoint. Phases:
1) Load skill/tools guidance
2) Discover path hints
3) Return metadata additions for downstream prompts
"""

# ---------------------------------------------------------------------------
# Section: parser helper
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Section: LLM extraction helper
# ---------------------------------------------------------------------------

def _extract_paths_with_llm(
    agent_key: str,
    skill_text: str,
    tools_text: str,
) -> tuple[list[str], list[str], list[str], list[str], dict[str, Any]]:
    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "lite")
    except Exception:
        return [], [], [], [], {}

    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing an agent's configuration to extract working file paths.\n\n"
                "## Agent Configuration\n\n"
                "### SKILL\n"
                f"{skill_text}\n\n"
                "### TOOLS\n"
                f"{tools_text}\n\n"
                "## Task\n"
                "Extract file and folder paths that this agent works with.\n"
                "Classify each path as source input path or output destination path.\n"
                "Return only JSON with this exact shape:\n"
                "{\"source_paths\":[\"path1\"],\"output_paths\":[\"path2\"],\"source_filenames\":[\"file1\"],\"output_filenames\":[\"file2\"]}\n\n"
                "## Rules\n"
                "- if example is given using {{}}, <>, or using example string for source_paths, output_paths, source_filenames, output_filenames replace them with actual information found in SKILL, TOOLS.\n\n"
                "- include only plausible file/folder paths\n"
                "- source_paths: folder paths where the agent reads input files\n"
                "- output_paths: folder paths where the agent writes generated files\n"
                "- source_filenames: actual source filenames the agent works with (not SKILL.md, TOOLS.md, or PERSONA.md)\n"
                "- output_filenames: actual output filenames or patterns the agent generates\n"
                "- no commentary"
            ),
        },
        {"role": "user", "content": "Extract the file and folder paths."},
    ]

    try:
        raw_output, metric_discovery = timed_generate(provider, messages, node_name="discovery")
    except Exception:
        return [], [], [], [], {}

    payload = _extract_json_object(str(raw_output or ""))
    if not isinstance(payload, dict):
        return [], [], [], [], metric_discovery

    source_raw = payload.get("source_paths")
    output_raw = payload.get("output_paths")
    source_filenames_raw = payload.get("source_filenames")
    output_filenames_raw = payload.get("output_filenames")
    # Backward compatibility for older format.
    legacy_paths = payload.get("paths")
    if not isinstance(source_raw, list) and isinstance(legacy_paths, list):
        source_raw = legacy_paths

    source_paths: list[str] = []
    output_paths: list[str] = []
    source_filenames: list[str] = []
    output_filenames: list[str] = []
    for value in source_raw if isinstance(source_raw, list) else []:
        if isinstance(value, str):
            path = value.strip().rstrip(".,;:)]")
            if path and path not in source_paths:
                source_paths.append(path)
    for value in output_raw if isinstance(output_raw, list) else []:
        if isinstance(value, str):
            path = value.strip().rstrip(".,;:)]")
            if path and path not in output_paths:
                output_paths.append(path)
    for value in source_filenames_raw if isinstance(source_filenames_raw, list) else []:
        if isinstance(value, str):
            filename = value.strip().rstrip(".,;:)]")
            if filename and filename not in source_filenames:
                source_filenames.append(filename)
    for value in output_filenames_raw if isinstance(output_filenames_raw, list) else []:
        if isinstance(value, str):
            filename = value.strip().rstrip(".,;:)]")
            if filename and filename not in output_filenames:
                output_filenames.append(filename)
    return source_paths, output_paths, source_filenames, output_filenames, metric_discovery


# ---------------------------------------------------------------------------
# Section: source-item mapping helpers
# ---------------------------------------------------------------------------


def _item_key_from_url(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"u_{digest}"


def _extract_urls_from_input(user_input: str) -> list[str]:
    if not isinstance(user_input, str) or not user_input.strip():
        return []
    pattern = re.compile(r"https?://[^\s)>\"']+")
    raw_urls = pattern.findall(user_input)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in raw_urls:
        normalized = url.strip().rstrip(".,;:!?)]")
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _extract_urls_from_chat_history(chat_history: list[dict[str, Any]] | None) -> list[str]:
    if not isinstance(chat_history, list) or not chat_history:
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for entry in chat_history:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("role") or "").strip() != "user":
            continue
        for url in _extract_urls_from_input(str(entry.get("content") or "")):
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _build_target_artifacts_source(user_input: str, chat_history: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    urls = _extract_urls_from_chat_history(chat_history)
    for url in _extract_urls_from_input(user_input):
        if url not in urls:
            urls.append(url)
    items: list[dict[str, Any]] = []
    for idx, url in enumerate(urls, start=1):
        items.append(
            {
                "index": idx,
                "source_type": "url",
                "source_ref": url,
                "item_key": _item_key_from_url(url),
            }
        )
    return items


# ---------------------------------------------------------------------------
# Section: core node
# ---------------------------------------------------------------------------


def discovery_node(state: AgentState) -> dict[str, Any]:
    # Phase 1: validate required state.
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {
            "source_paths": [],
            "output_paths": [],
            "source_filenames": [],
            "output_filenames": [],
            "metric_discovery": {},
        }

    # Phase 2: load agent static guidance (best effort).
    try:
        skill_text = load_agent_skill(agent_key)
    except Exception:
        skill_text = ""
    try:
        tools_text = load_agent_tools(agent_key)
    except Exception:
        tools_text = ""

    # Phase 3: extract path hints using discovery model call.
    source_paths, output_paths, source_filenames, output_filenames, metric_discovery = _extract_paths_with_llm(
        agent_key,
        skill_text,
        tools_text,
    )
    # Phase 4: merge discovery artifacts into metadata.
    metadata = dict(state.get("metadata", {}))
    target_artifacts_source = _build_target_artifacts_source(
        str(state.get("user_input") or ""),
        load_chat_history(state.get("conversation_id") or state.get("thread_id"), limit=12),
    )
    metadata["source_paths"] = source_paths
    metadata["output_paths"] = output_paths
    metadata["source_filenames"] = source_filenames
    metadata["output_filenames"] = output_filenames
    metadata["target_artifacts_source"] = target_artifacts_source

    return {
        "source_paths": source_paths,
        "output_paths": output_paths,
        "source_filenames": source_filenames,
        "output_filenames": output_filenames,
        "target_artifacts_source": target_artifacts_source,
        "metadata": metadata,
        **build_metric_state_delta("discovery", "metric_discovery", metric_discovery),
    }
