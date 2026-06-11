"""Agent endpoints — wraps ``solidcue.services.agent_service`` and ``thread_service``."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import AsyncIterator

import yaml
from dotenv import set_key
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from solidcue.agents.configs.loader import get_agent_path
from solidcue.agents.configs.schema import AgentConfig, ProviderConfig
from solidcue.api.run_result import build_run_response
from solidcue.api.schemas import (
    ResumeAgentRequest,
    RunAgentRequest,
    RunAgentResponse,
    StreamAgentRequest,
    ThreadResponse,
)
from solidcue.observability import generate_env_key
from solidcue.observability.env import get_env_path
from solidcue.services.agent_service import (
    CreateAgentInput,
    create_agent,
    get_agents,
)
from solidcue.services.run_engine import (
    cancel_run,
    iter_run_events,
    run_agent_step,
    start_run,
)
from solidcue.services.state_snapshot_service import get_latest_thread_id_for_conversation
from solidcue.services.thread_service import create_thread_id


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


def _sse_frame(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_agent_sse(
    *,
    agent_key: str,
    thread_id: str | None,
    conversation_id: str | None = None,
    user_input: str | None = None,
    resume_value: str | None = None,
) -> AsyncIterator[str]:
    run_id = await start_run(
        agent_key=agent_key,
        thread_id=thread_id,
        conversation_id=conversation_id,
        user_input=user_input,
        resume_value=resume_value,
    )
    async for event in iter_run_events(run_id):
        name = event.get("event")
        data = event.get("data")
        if isinstance(name, str) and isinstance(data, dict):
            yield _sse_frame(name, data)


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


@router.post("/threads", response_model=ThreadResponse)
def new_thread() -> ThreadResponse:
    return ThreadResponse(thread_id=create_thread_id())


@router.post("/{agent_key}/run", response_model=RunAgentResponse)
def run(agent_key: str, request: RunAgentRequest) -> RunAgentResponse:
    thread_id = request.thread_id or create_thread_id()
    try:
        _agent, result = run_agent_step(
            agent_key=agent_key,
            thread_id=thread_id,
            user_input=request.user_input,
            debug=request.debug,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return build_run_response(thread_id=thread_id, result=result)


@router.post("/{agent_key}/resume", response_model=RunAgentResponse)
def resume(agent_key: str, request: ResumeAgentRequest) -> RunAgentResponse:
    try:
        _agent, result = run_agent_step(
            agent_key=agent_key,
            thread_id=request.thread_id,
            resume_value=request.resume_value,
            debug=request.debug,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return build_run_response(thread_id=request.thread_id, result=result)


@router.post("/{agent_key}/stream")
async def stream(agent_key: str, request: StreamAgentRequest) -> StreamingResponse:
    """Stream a run (or resume) as Server-Sent Events.

    Uses POST (not a GET EventSource) so the run/resume body travels in the
    request; the frontend reads the SSE frames from the response stream.
    """
    _validate_key(agent_key)
    is_continue = request.resume_value is None and request.user_input is None
    if is_continue and not request.thread_id:
        if not request.conversation_id:
            raise HTTPException(status_code=400, detail="user_input, resume_value, thread_id, or conversation_id is required")
    if request.resume_value is not None and not request.thread_id and not request.conversation_id:
        raise HTTPException(status_code=400, detail="thread_id or conversation_id is required to resume")

    # Validate the agent exists up front so a missing agent is a clean 404,
    # before we switch into a 200 streaming response.
    if not get_agent_path(agent_key).exists():
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_key}")

    resolved_thread_id = request.thread_id
    if not resolved_thread_id and request.conversation_id:
        resolved_thread_id = get_latest_thread_id_for_conversation(request.conversation_id)
    if not resolved_thread_id and request.resume_value is None:
        resolved_thread_id = create_thread_id()
    if request.resume_value is not None and not resolved_thread_id:
        raise HTTPException(status_code=400, detail="No existing thread found for conversation_id")
    generator = _stream_agent_sse(
        agent_key=agent_key,
        thread_id=resolved_thread_id,
        conversation_id=request.conversation_id,
        user_input=request.user_input,
        resume_value=request.resume_value,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{agent_key}/runs/{run_id}/cancel", status_code=200)
def cancel(agent_key: str, run_id: str) -> dict:
    """Cancel an active run by run_id.  Idempotent — safe to call on completed runs."""
    cancelled = cancel_run(run_id)
    return {"run_id": run_id, "cancelled": cancelled}


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
