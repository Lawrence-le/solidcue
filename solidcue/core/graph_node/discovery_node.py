from __future__ import annotations

import json
import re
from typing import Any

from solidcue.agents.configs.loader import load_agent, load_agent_persona
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState

_DRIVE_PATH_PATTERN = re.compile(r'Google Drive path\s+"([^"]+)"', re.IGNORECASE)


def _extract_persona_source_paths(persona_text: str) -> list[str]:
    if not persona_text:
        return []

    found: list[str] = []
    for match in _DRIVE_PATH_PATTERN.finditer(persona_text):
        candidate = match.group(1).strip()
        if candidate and candidate not in found:
            found.append(candidate)
    return found


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


def _extract_persona_source_paths_with_llm(agent_key: str, persona_text: str) -> list[str]:
    try:
        agent = load_agent(agent_key)
        provider = get_provider_for_role(agent, "decision")
    except Exception:
        return []

    messages = [
        {
            "role": "system",
            "content": (
                "Extract Google Drive source paths from persona text.\n"
                "Return only JSON with this shape: {\"paths\":[\"path1\",\"path2\"]}.\n"
                "Rules: include only plausible Google Drive file/folder paths; no commentary."
            ),
        },
        {"role": "user", "content": persona_text},
    ]

    try:
        raw_output = provider.generate(messages)
    except Exception:
        return []

    payload = _extract_json_object(str(raw_output or ""))
    if not isinstance(payload, dict):
        return []

    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list):
        return []

    normalized: list[str] = []
    for value in raw_paths:
        if not isinstance(value, str):
            continue
        path = value.strip()
        if path and path not in normalized:
            normalized.append(path)
    return normalized


def discovery_node(state: AgentState) -> dict[str, Any]:
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {"persona_source_paths": []}

    try:
        persona_text = load_agent_persona(agent_key)
    except Exception:
        return {"persona_source_paths": []}

    source_paths = _extract_persona_source_paths(persona_text)
    if not source_paths:
        source_paths = _extract_persona_source_paths_with_llm(agent_key, persona_text)
    metadata = dict(state.get("metadata", {}))
    metadata["persona_source_paths"] = source_paths

    return {
        "persona_source_paths": source_paths,
        "metadata": metadata,
    }
