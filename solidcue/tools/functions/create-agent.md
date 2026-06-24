# Function Reference

# Create Agent Functions

## Purpose

Reference for the functions and side effects required to create a finalized
agent folder and config.

## Required Functions

### `create_agent(input_data: CreateAgentInput) -> tuple[AgentConfig, str]`

Creates the finalized agent configuration and writes it to disk.

Responsibilities:

- build provider configs
- write environment key values
- assemble the `AgentConfig`
- save the YAML config
- save the default agent markdown files

### `generate_env_key(prefix: str) -> str`

Build the environment variable name for an agent provider key.

### `write_env_key(env_key: str, value: str) -> None`

Persist the API key for a provider into the workspace environment file.

### `save_agent(config: AgentConfig) -> Path`

Write the agent YAML config to `solidcue/agents/<agent_key>/<agent_key>.yaml`.

### `save_agent_persona(agent_key: str, content: str | None = None) -> Path`

Write the agent `PERSONA.md`.

### `save_agent_skill(agent_key: str, content: str | None = None) -> Path`

Write the agent `SKILL.md`.

### `save_agent_tools(agent_key: str, content: str | None = None) -> Path`

Write the agent `TOOLS.md`.

## Supporting Functions

- `get_agent_path(agent_key: str)`
- `load_agent(agent_key: str)`
- `list_agents()`

These functions are used for validation and lookup, not for writing the final
artifact.

## Input Mapping

- agent name -> `CreateAgentInput.name`
- agent key -> `CreateAgentInput.agent_key`
- description -> `CreateAgentInput.description`
- decision provider -> `CreateAgentInput.decision_*`
- lite provider -> `CreateAgentInput.lite_*`
- reviewer provider -> `CreateAgentInput.reviewer_*`
- writer provider -> `CreateAgentInput.writer_*`
- selected tools -> `CreateAgentInput.selected_tools`

## Side Effects

- Write provider API keys to the environment store.
- Create the agent folder structure.
- Create the agent config YAML.
- Create default `PERSONA.md`, `SKILL.md`, and `TOOLS.md`.

## Validation Rules

- Never overwrite an existing agent.
- Never write incomplete provider configuration.
- Never generate files outside `solidcue/agents/<agent_key>/`.
- Fail fast if the agent key already exists.
