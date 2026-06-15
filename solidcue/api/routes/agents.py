"""Agent endpoints — wraps ``solidcue.services.agent_service`` and ``thread_service``."""

from __future__ import annotations

import re
import shutil

import yaml
from dotenv import set_key
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from solidcue.agent_configs.loader import get_agent_path
from solidcue.agent_configs.schema import AgentConfig, ProviderConfig
from solidcue.observability import generate_env_key
from solidcue.observability.env import get_env_path
from solidcue.services.agent_service import CreateAgentInput, create_agent
from solidcue.services.workspace_service import get_agents


class UpdateAgentInput(BaseModel):
    name: str
    description: str
    decision_provider_type: str
    decision_base_url: str | None = None
    decision_api_key: str | None = None  # blank = keep existing
    decision_model: str
    decision_temperature: float
    lite_provider_type: str
    lite_base_url: str | None = None
    lite_api_key: str | None = None
    lite_model: str
    lite_temperature: float
    reviewer_provider_type: str
    reviewer_base_url: str | None = None
    reviewer_api_key: str | None = None
    reviewer_model: str
    reviewer_temperature: float
    writer_provider_type: str | None = None
    writer_base_url: str | None = None
    writer_api_key: str | None = None
    writer_model: str | None = None
    writer_temperature: float | None = None
    selected_tools: list[str]


def _set_env_key_if_provided(env_key: str, value: str | None) -> None:
    if not value or not value.strip():
        return
    env_path = get_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not env_path.exists():
        env_path.touch()
    set_key(str(env_path), env_key, value.strip(), quote_mode="never")


def _make_provider(
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

router = APIRouter(prefix="/agents", tags=["agents"])

_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_key(agent_key: str) -> None:
    if not _KEY_PATTERN.match(agent_key):
        raise HTTPException(status_code=400, detail=f"Invalid agent key: {agent_key}")


@router.get("", response_model=list[AgentConfig])
def list_agents() -> list[AgentConfig]:
    return get_agents()


@router.post("", response_model=AgentConfig, status_code=201)
def create(input_data: CreateAgentInput) -> AgentConfig:
    try:
        config, _path = create_agent(input_data)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return config


@router.put("/{agent_key}", response_model=AgentConfig)
def update(agent_key: str, input_data: UpdateAgentInput) -> AgentConfig:
    _validate_key(agent_key)
    agent_path = get_agent_path(agent_key)
    if not agent_path.exists():
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_key}")

    brain_env = generate_env_key(f"{agent_key}_brain")
    lite_env = generate_env_key(f"{agent_key}_lite")
    reviewer_env = generate_env_key(f"{agent_key}_reviewer")
    writer_env = generate_env_key(f"{agent_key}_writer")

    _set_env_key_if_provided(brain_env, input_data.decision_api_key)
    _set_env_key_if_provided(lite_env, input_data.lite_api_key)
    _set_env_key_if_provided(reviewer_env, input_data.reviewer_api_key)
    _set_env_key_if_provided(writer_env, input_data.writer_api_key)

    writer_provider = None
    if input_data.writer_provider_type and input_data.writer_model:
        writer_provider = _make_provider(
            input_data.writer_provider_type,
            input_data.writer_base_url,
            writer_env,
            input_data.writer_model,
            input_data.writer_temperature,
        )

    config = AgentConfig(
        agent_key=agent_key,
        name=input_data.name,
        description=input_data.description,
        provider=_make_provider(
            input_data.decision_provider_type,
            input_data.decision_base_url,
            brain_env,
            input_data.decision_model,
            input_data.decision_temperature,
        ),
        lite_provider=_make_provider(
            input_data.lite_provider_type,
            input_data.lite_base_url,
            lite_env,
            input_data.lite_model,
            input_data.lite_temperature,
        ),
        reviewer_provider=_make_provider(
            input_data.reviewer_provider_type,
            input_data.reviewer_base_url,
            reviewer_env,
            input_data.reviewer_model,
            input_data.reviewer_temperature,
        ),
        writer_provider=writer_provider,
        tools=input_data.selected_tools,
    )

    with agent_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)

    return config


@router.delete("/{agent_key}", status_code=204)
def delete(agent_key: str) -> Response:
    _validate_key(agent_key)
    agent_dir = get_agent_path(agent_key).parent
    if not agent_dir.exists():
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_key}")
    shutil.rmtree(agent_dir)
    return Response(status_code=204)
