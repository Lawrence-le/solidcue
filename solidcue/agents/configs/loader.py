from pathlib import Path

import yaml

from solidcue.agents.configs.schema import AgentConfig

AGENTS_ROOT_DIR = Path(__file__).parent.parent
LEGACY_AGENT_CONFIG_DIR = AGENTS_ROOT_DIR / "configs"
DEFAULT_PERSONA_TEMPLATE = """# Persona

## Role
Describe this agent's role and responsibilities.

## Communication Style
- Tone:
- Level of detail:
- Formatting preferences:

## Behavioral Rules
- Prioritize:
- Avoid:
- Escalation policy:

## Domain Preferences
- Preferred sources or tools:
- Domain constraints:
"""


def get_agent_path(agent_key: str) -> Path:
    return AGENTS_ROOT_DIR / agent_key / f"{agent_key}.yaml"


def get_persona_path(agent_key: str) -> Path:
    return AGENTS_ROOT_DIR / agent_key / "PERSONA.md"


def _get_legacy_agent_path(agent_key: str) -> Path:
    return LEGACY_AGENT_CONFIG_DIR / f"{agent_key}.yaml"


def _get_legacy_persona_path(agent_key: str) -> Path:
    return LEGACY_AGENT_CONFIG_DIR / f"{agent_key}.PERSONA.md"


def save_agent_persona(agent_key: str, content: str | None = None) -> Path:
    path = get_persona_path(agent_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(f"Agent persona already exists: {agent_key}")

    persona_content = content if content is not None else DEFAULT_PERSONA_TEMPLATE
    with path.open("w", encoding="utf-8") as file:
        file.write(persona_content.rstrip() + "\n")

    return path


def load_agent_persona(agent_key: str) -> str:
    path = get_persona_path(agent_key)
    if not path.exists():
        legacy_path = _get_legacy_persona_path(agent_key)
        if not legacy_path.exists():
            return ""
        path = legacy_path

    content = path.read_text(encoding="utf-8").strip()
    return content


def save_agent(config: AgentConfig) -> Path:
    path = get_agent_path(config.agent_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(f"Agent key already exists: {config.agent_key}")

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.model_dump(), file, sort_keys=False)

    return path


def load_agent(agent_key: str) -> AgentConfig:
    path = get_agent_path(agent_key)
    if not path.exists():
        legacy_path = _get_legacy_agent_path(agent_key)
        if not legacy_path.exists():
            raise FileNotFoundError(f"Agent not found: {agent_key}")
        path = legacy_path

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Agent configuration is empty: {agent_key}")

    return AgentConfig(**data)


def list_agents() -> list[AgentConfig]:
    candidate_paths: list[Path] = []
    if AGENTS_ROOT_DIR.exists():
        candidate_paths.extend(
            path
            for path in AGENTS_ROOT_DIR.glob("*/*.yaml")
            if path.parent.name not in {"__pycache__", "configs"}
        )
    if LEGACY_AGENT_CONFIG_DIR.exists():
        candidate_paths.extend(LEGACY_AGENT_CONFIG_DIR.glob("*.yaml"))

    agents: list[AgentConfig] = []
    for path in candidate_paths:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if data is None:
            continue

        agents.append(AgentConfig(**data))

    return agents
