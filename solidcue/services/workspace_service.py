"""Workspace-level helpers for agent discovery and no-agent-key flows.

This module is intended for setup/bootstrap workflows that operate before a
specific agent is selected or created.
"""

from __future__ import annotations

from pathlib import Path

from solidcue.agent_configs.loader import AGENTS_ROOT_DIR, SKILLS_ROOT_DIR, list_agents, load_agent
from solidcue.agent_configs.schema import AgentConfig


def get_workspace_root() -> Path:
    return AGENTS_ROOT_DIR


def list_workspace_agents() -> list[AgentConfig]:
    return list_agents()


def get_agents() -> list[AgentConfig]:
    return list_workspace_agents()


def list_agent_keys() -> list[str]:
    return [agent.agent_key for agent in list_workspace_agents() if agent.agent_key.strip()]


def has_agents() -> bool:
    return any(list_agent_keys())


def agent_exists(agent_key: str) -> bool:
    try:
        load_agent(agent_key)
    except FileNotFoundError:
        return False
    return True


def load_agent_if_exists(agent_key: str) -> AgentConfig | None:
    try:
        return load_agent(agent_key)
    except FileNotFoundError:
        return None


def get_skills_root() -> Path:
    return SKILLS_ROOT_DIR


def get_create_agent_skill_path() -> Path:
    return SKILLS_ROOT_DIR / "create-agent.md"


def load_create_agent_skill() -> str:
    path = get_create_agent_skill_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def resolve_system_skill_key(system_intent: str) -> str:
    normalized = str(system_intent or "").strip().casefold().replace("_", "-")
    if normalized in {"create-agent", "create agent", "bootstrap"}:
        return "create-agent"
    if normalized in {"create-skill", "create skill", "skill"}:
        return "create-skill"
    if normalized in {"setup-provider", "provider", "repair-config", "import-agent", "select-agent"}:
        return "user-profile"
    return "user-profile"


def get_system_skill_path(system_intent: str) -> Path:
    skill_key = resolve_system_skill_key(system_intent)
    return SKILLS_ROOT_DIR / f"{skill_key}.md"


def list_system_skill_keys() -> list[str]:
    if not SKILLS_ROOT_DIR.exists():
        return []
    keys: list[str] = []
    for path in sorted(SKILLS_ROOT_DIR.glob("*.md")):
        if path.is_file():
            keys.append(path.stem)
    return keys


def load_system_skill(system_intent: str) -> str:
    path = get_system_skill_path(system_intent)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()
