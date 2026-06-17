from __future__ import annotations

import json
from typing import Any

from solidcue.agent_configs.loader import load_agent, load_agent_skill, load_agent_tools, get_discovery_path
from solidcue.providers.provider_resolver import get_provider_for_role
from solidcue.core.graph_agent.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_async_stream_generate
from solidcue.core.utils.source_extraction import build_target_artifacts_source

"""
Discovery Node - Function Overview
----------------------------------

_extract_json_object:
Parse discovery/model JSON payloads robustly.

_load_discovery_cache:
Load discovery.json from agent folder if it exists.

_save_discovery_cache:
Write discovery result to discovery.json in agent folder.

_extract_paths_with_llm:
Extract path/filename hints from SKILL/TOOLS guidance. Called only when
discovery.json is absent; result is saved for subsequent runs.

discovery_node:
Main entrypoint. Phases:
1) Load discovery.json if present (cache hit — no LLM call)
2) If missing, call LLM and save result to discovery.json (cache miss)
3) Resolve target_artifacts_source from state or user_input fallback
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
# Section: discovery cache helpers
# ---------------------------------------------------------------------------


def _load_discovery_cache(agent_key: str) -> dict[str, Any] | None:
    path = get_discovery_path(agent_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_discovery_cache(agent_key: str, data: dict[str, Any]) -> None:
    path = get_discovery_path(agent_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _parse_path_list(raw: Any) -> list[str]:
    result: list[str] = []
    for value in raw if isinstance(raw, list) else []:
        if isinstance(value, str):
            cleaned = value.strip().rstrip(".,;:)]")
            if cleaned and cleaned not in result:
                result.append(cleaned)
    return result


# ---------------------------------------------------------------------------
# Section: LLM extraction helper
# ---------------------------------------------------------------------------

async def _extract_paths_with_llm(
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
        raw_output, metric_discovery = await timed_async_stream_generate(provider, messages, node_name="discovery")
    except Exception:
        return [], [], [], [], {}

    payload = _extract_json_object(str(raw_output or ""))
    if not isinstance(payload, dict):
        return [], [], [], [], metric_discovery

    source_raw = payload.get("source_paths")
    # Backward compatibility for older format.
    legacy_paths = payload.get("paths")
    if not isinstance(source_raw, list) and isinstance(legacy_paths, list):
        source_raw = legacy_paths

    source_paths = _parse_path_list(source_raw)
    output_paths = _parse_path_list(payload.get("output_paths"))
    source_filenames = _parse_path_list(payload.get("source_filenames"))
    output_filenames = _parse_path_list(payload.get("output_filenames"))

    return source_paths, output_paths, source_filenames, output_filenames, metric_discovery


# ---------------------------------------------------------------------------
# Section: core node
# ---------------------------------------------------------------------------


async def discovery_node(state: AgentState) -> dict[str, Any]:
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

    # Phase 2: load from discovery.json cache if present.
    metric_discovery: dict[str, Any] = {}
    cached = _load_discovery_cache(agent_key)
    if cached is not None:
        source_paths = _parse_path_list(cached.get("source_paths"))
        output_paths = _parse_path_list(cached.get("output_paths"))
        source_filenames = _parse_path_list(cached.get("source_filenames"))
        output_filenames = _parse_path_list(cached.get("output_filenames"))
    else:
        # Phase 3: cache miss — call LLM and save result to discovery.json.
        try:
            skill_text = load_agent_skill(agent_key)
        except Exception:
            skill_text = ""
        try:
            tools_text = load_agent_tools(agent_key)
        except Exception:
            tools_text = ""

        source_paths, output_paths, source_filenames, output_filenames, metric_discovery = await _extract_paths_with_llm(
            agent_key,
            skill_text,
            tools_text,
        )
        _save_discovery_cache(agent_key, {
            "source_paths": source_paths,
            "output_paths": output_paths,
            "source_filenames": source_filenames,
            "output_filenames": output_filenames,
        })

    # Phase 4: resolve target_artifacts_source — use router-populated value if present,
    # otherwise fall back to extracting from user_input (direct agent invocation).
    metadata = dict(state.get("metadata", {}))
    target_artifacts_source = state.get("target_artifacts_source") or build_target_artifacts_source(
        str(state.get("user_input") or ""),
        state.get("chat_history") or [],
    )

    return {
        "source_paths": source_paths,
        "output_paths": output_paths,
        "source_filenames": source_filenames,
        "output_filenames": output_filenames,
        "target_artifacts_source": target_artifacts_source,
        "metadata": metadata,
        **build_metric_state_delta("discovery", "metric_discovery", metric_discovery),
    }
