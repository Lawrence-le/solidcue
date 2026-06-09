"""Agent CRUD — create, list, and persist agent configurations.

Execution (run, stream, resume) lives in ``solidcue.services.run_engine``.
"""

from pydantic import BaseModel

from solidcue.agents.configs.loader import (
    list_agents,
    load_agent,
    save_agent,
    save_agent_persona,
    save_agent_skill,
    save_agent_tools,
)
from solidcue.agents.configs.schema import AgentConfig, ProviderConfig
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


def create_agent(input_data: CreateAgentInput) -> tuple[AgentConfig, str]:
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
    )
    path = save_agent(config)
    save_agent_persona(config.agent_key)
    save_agent_skill(config.agent_key)
    save_agent_tools(config.agent_key)
    return config, str(path)


def get_agents() -> list[AgentConfig]:
    return list_agents()
