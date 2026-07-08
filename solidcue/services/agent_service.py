"""Agent CRUD — create, list, and persist agent configurations.

Execution (run, stream, resume) lives in ``solidcue.services.run_engine``.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from solidcue.agent_configs.loader import (
    save_agent,
    save_agent_persona,
    save_agent_skill,
    save_agent_tools,
)
from solidcue.agent_configs.schema import AgentConfig, PlanningPolicy, ProviderConfig
from solidcue.observability import generate_env_key, write_env_key


class CreateAgentInput(BaseModel):
    name: str
    agent_key: str
    description: str
    decision_provider_type: str
    decision_base_url: str | None
    decision_api_key: str
    decision_model: str
    decision_temperature: float
    lite_provider_type: str
    lite_base_url: str | None
    lite_api_key: str
    lite_model: str
    lite_temperature: float
    reviewer_provider_type: str
    reviewer_base_url: str | None
    reviewer_api_key: str
    reviewer_model: str
    reviewer_temperature: float
    writer_provider_type: str | None = None
    writer_base_url: str | None = None
    writer_api_key: str | None = None
    writer_model: str | None = None
    writer_temperature: float | None = None
    selected_tools: list[str]
    planning_mode: Literal["static", "dynamic"] = "dynamic"

    # Remaining AgentConfig fields — optional so CLI/REST callers keep working,
    # but now part of the contract so collect_spec can gather them and the YAML
    # stops defaulting them silently.
    allowed_tasks: list[str] = Field(default_factory=list)
    style: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    validation_policy: str | None = None

    # Definition (.md) substance — fields the writer needs but the YAML has no
    # home for. Gathered from the human, passed to graph_definition so PERSONA/
    # SKILL/TOOLS are grounded in real intent instead of improvised.
    produces_artifacts: bool | None = None
    artifact_destination: str | None = None
    key_tasks: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("selected_tools", "key_tasks", "allowed_tasks", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: Any) -> Any:
        """Accept a string where a list is expected — LLM-driven callers (the
        router) often emit "a, b, c" or a newline block instead of a JSON array.
        Split it into a list rather than raising a ValidationError that aborts
        agent creation."""
        if value is None:
            return []
        if isinstance(value, str):
            parts = re.split(r"[\n;,]", value)
            return [p.strip() for p in parts if p.strip()]
        return value

    @field_validator("examples", mode="before")
    @classmethod
    def _coerce_examples(cls, value: Any) -> Any:
        """Keep only well-formed dict examples; drop a stray string/None the model
        might emit instead of failing the whole spec."""
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]


def _build_provider_config(
    *,
    provider_type: str,
    base_url: str | None,
    api_key_env: str,
    model: str,
    temperature: float | None,
) -> ProviderConfig:
    return ProviderConfig(
        type=provider_type,
        base_url=base_url or None,
        api_key_env=api_key_env,
        model=model,
        temperature=temperature,
    )


def _build_agent_config_and_env(input_data: "CreateAgentInput") -> tuple[AgentConfig, str, str, str, str]:
    """Build AgentConfig and write env keys. Returns (config, brain_env, lite_env, reviewer_env, writer_env)."""
    brain_env_key = generate_env_key(f"{input_data.agent_key}_brain")
    lite_env_key = generate_env_key(f"{input_data.agent_key}_lite")
    reviewer_env_key = generate_env_key(f"{input_data.agent_key}_reviewer")
    writer_env_key = generate_env_key(f"{input_data.agent_key}_writer")
    write_env_key(brain_env_key, input_data.decision_api_key)
    write_env_key(lite_env_key, input_data.lite_api_key)
    write_env_key(reviewer_env_key, input_data.reviewer_api_key)
    if input_data.writer_api_key:
        write_env_key(writer_env_key, input_data.writer_api_key)

    brain_provider = _build_provider_config(
        provider_type=input_data.decision_provider_type,
        base_url=input_data.decision_base_url,
        api_key_env=brain_env_key,
        model=input_data.decision_model,
        temperature=input_data.decision_temperature,
    )
    lite_provider = _build_provider_config(
        provider_type=input_data.lite_provider_type,
        base_url=input_data.lite_base_url,
        api_key_env=lite_env_key,
        model=input_data.lite_model,
        temperature=input_data.lite_temperature,
    )
    reviewer_provider = _build_provider_config(
        provider_type=input_data.reviewer_provider_type,
        base_url=input_data.reviewer_base_url,
        api_key_env=reviewer_env_key,
        model=input_data.reviewer_model,
        temperature=input_data.reviewer_temperature,
    )
    writer_provider = None
    if (
        input_data.writer_provider_type
        and input_data.writer_model
        and input_data.writer_api_key
    ):
        writer_provider = _build_provider_config(
            provider_type=input_data.writer_provider_type,
            base_url=input_data.writer_base_url,
            api_key_env=writer_env_key,
            model=input_data.writer_model,
            temperature=input_data.writer_temperature,
        )

    config = AgentConfig(
        agent_key=input_data.agent_key,
        name=input_data.name,
        description=input_data.description,
        provider=brain_provider,
        lite_provider=lite_provider,
        reviewer_provider=reviewer_provider,
        writer_provider=writer_provider,
        tools=input_data.selected_tools,
        planning=PlanningPolicy(mode=input_data.planning_mode),
        allowed_tasks=input_data.allowed_tasks,
        style=input_data.style,
        constraints=input_data.constraints,
        validation_policy=input_data.validation_policy,
    )
    return config, brain_env_key, lite_env_key, reviewer_env_key, writer_env_key


def write_agent_config(input_data: "CreateAgentInput") -> tuple[AgentConfig, str]:
    """Write env keys and YAML config only — no MD files. Called by graph_system write_config_node."""
    config, *_ = _build_agent_config_and_env(input_data)
    path = save_agent(config)
    return config, str(path)


def create_agent(input_data: "CreateAgentInput") -> tuple[AgentConfig, str]:
    """Write env keys, YAML config, and default MD templates. Called by REST/CLI paths."""
    config, *_ = _build_agent_config_and_env(input_data)
    path = save_agent(config)
    save_agent_persona(config.agent_key)
    save_agent_skill(config.agent_key)
    save_agent_tools(config.agent_key)
    return config, str(path)
