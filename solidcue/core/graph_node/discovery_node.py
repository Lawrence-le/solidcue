from __future__ import annotations

import json
from typing import Any

from solidcue.agents.configs.loader import load_agent, load_agent_skill, load_agent_tools
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate


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
        raw_output, metric_discovery = timed_generate(provider, messages)
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


def discovery_node(state: AgentState) -> dict[str, Any]:
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {
            "source_paths": [],
            "output_paths": [],
            "source_filenames": [],
            "output_filenames": [],
            "metric_discovery": {},
        }

    try:
        skill_text = load_agent_skill(agent_key)
    except Exception:
        skill_text = ""
    try:
        tools_text = load_agent_tools(agent_key)
    except Exception:
        tools_text = ""

    source_paths, output_paths, source_filenames, output_filenames, metric_discovery = _extract_paths_with_llm(
        agent_key,
        skill_text,
        tools_text,
    )
    metadata = dict(state.get("metadata", {}))
    metadata["source_paths"] = source_paths
    metadata["output_paths"] = output_paths
    metadata["source_filenames"] = source_filenames
    metadata["output_filenames"] = output_filenames

    return {
        "source_paths": source_paths,
        "output_paths": output_paths,
        "source_filenames": source_filenames,
        "output_filenames": output_filenames,
        "metadata": metadata,
        **build_metric_state_delta("discovery", "metric_discovery", metric_discovery),
    }
