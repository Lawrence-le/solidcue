from pathlib import Path

import yaml

from solidcue.agent_configs.schema import AgentConfig

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
AGENTS_ROOT_DIR = PACKAGE_ROOT / "agents"
SKILLS_ROOT_DIR = PACKAGE_ROOT / "skills"
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
DEFAULT_TOOLS_TEMPLATE = """# Tools

Use this file to describe which tools this agent should use and how to choose between them.

## Tool Routing
- Use search/list tools when the exact file or resource path is unknown.
- Use read tools before making edits or generating outputs.
- Use write/edit tools only on approved source files.
- Use export/generation tools only after content edits are complete.

## Failure Handling
1. Read the tool error.
2. Try the safest fallback if one is available.
3. Ask the user before taking risky or destructive actions.

## Tool Notes
- Tool:
  - Use when:
  - Avoid when:
- Expected input:
  - Expected output:
"""


def _load_text_asset(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    content = path.read_text(encoding="utf-8").strip()
    return content or fallback


DEFAULT_SKILL_TEMPLATE = """# Skill

Use this file to describe the agent's workflow rules, source-of-truth files, and task-specific behavior.

## When To Use
- Use this skill when:
- Do not use this skill when:

## Information Sources
- Primary source:
- Supporting sources:
- Generated outputs:

## Workflow
1. Identify the user's goal.
2. Locate and read the required source files.
3. Use only grounded information from approved sources.
4. Complete the task.
5. Verify the result before responding.

## Rules
- Do not invent facts, paths, or outputs.
- Prefer simple, maintainable steps.
- Ask the user when required source information is missing.

## Verification
- Confirm source files were found and used.
- Confirm generated or edited outputs are in the expected location.
- Confirm the final answer is concise and accurate.
"""


def get_agent_path(agent_key: str) -> Path:
    return AGENTS_ROOT_DIR / agent_key / f"{agent_key}.yaml"


def get_persona_path(agent_key: str) -> Path:
    return AGENTS_ROOT_DIR / agent_key / "PERSONA.md"


def get_skill_path(agent_key: str) -> Path:
    return AGENTS_ROOT_DIR / agent_key / "SKILL.md"


def get_tools_path(agent_key: str) -> Path:
    return AGENTS_ROOT_DIR / agent_key / "TOOLS.md"


def _get_legacy_agent_path(agent_key: str) -> Path:
    return LEGACY_AGENT_CONFIG_DIR / f"{agent_key}.yaml"


def _get_legacy_persona_path(agent_key: str) -> Path:
    return LEGACY_AGENT_CONFIG_DIR / f"{agent_key}.PERSONA.md"


def _save_agent_markdown(
    *,
    agent_key: str,
    path: Path,
    content: str | None,
    default_content: str,
    label: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        raise FileExistsError(f"Agent {label} already exists: {agent_key}")

    markdown_content = content if content is not None else default_content
    with path.open("w", encoding="utf-8") as file:
        file.write(markdown_content.rstrip() + "\n")

    return path


def save_agent_persona(agent_key: str, content: str | None = None) -> Path:
    return _save_agent_markdown(
        agent_key=agent_key,
        path=get_persona_path(agent_key),
        content=content,
        default_content=DEFAULT_PERSONA_TEMPLATE,
        label="persona",
    )


def save_agent_skill(agent_key: str, content: str | None = None) -> Path:
    return _save_agent_markdown(
        agent_key=agent_key,
        path=get_skill_path(agent_key),
        content=content,
        default_content=DEFAULT_SKILL_TEMPLATE,
        label="skill",
    )


def save_agent_tools(agent_key: str, content: str | None = None) -> Path:
    return _save_agent_markdown(
        agent_key=agent_key,
        path=get_tools_path(agent_key),
        content=content,
        default_content=DEFAULT_TOOLS_TEMPLATE,
        label="tools",
    )


def load_agent_persona(agent_key: str) -> str:
    path = get_persona_path(agent_key)
    if not path.exists():
        legacy_path = _get_legacy_persona_path(agent_key)
        if not legacy_path.exists():
            return ""
        path = legacy_path

    content = path.read_text(encoding="utf-8").strip()
    return content


def load_agent_skill(agent_key: str) -> str:
    path = get_skill_path(agent_key)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_agent_tools(agent_key: str) -> str:
    path = get_tools_path(agent_key)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


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


def get_create_agent_skill_path() -> Path:
    return SKILLS_ROOT_DIR / "create-agent.md"


def load_create_agent_skill() -> str:
    return _load_text_asset(get_create_agent_skill_path(), "")
