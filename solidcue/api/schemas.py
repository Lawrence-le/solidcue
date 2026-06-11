"""HTTP request/response models for the API layer.

These models exist only to shape JSON at the transport boundary. Wherever a
service already defines an input model (e.g. ``CreateAgentInput``,
``CreateMcpToolInput``), the routes reuse it directly rather than redefining it
here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunAgentRequest(BaseModel):
    """Start a new agent run on a fresh or supplied thread."""

    user_input: str
    thread_id: str | None = None
    debug: bool = False


class ResumeAgentRequest(BaseModel):
    """Resume an interrupted run with the user's approval/answer."""

    thread_id: str
    resume_value: str
    debug: bool = False


class RunAgentResponse(BaseModel):
    """Outcome of a single run/resume step.

    ``status`` is ``"interrupted"`` when the graph paused for approval — the
    caller should present ``interrupt`` to the user and POST the answer back to
    the resume endpoint. ``status`` is ``"completed"`` once a final output is
    produced.
    """

    thread_id: str
    status: Literal["completed", "interrupted"]
    output: str | None = None
    phase: str | None = None
    current_task: str | None = None
    interrupt: dict[str, Any] | None = None


class StreamAgentRequest(BaseModel):
    """Start or resume a streamed (SSE) run.

    Provide ``user_input`` to start a new run, or ``resume_value`` (with
    ``thread_id``) to answer an approval interrupt and continue.
    """

    thread_id: str | None = None
    conversation_id: str | None = None
    user_input: str | None = None
    resume_value: str | None = None


class RuntimeProviderRequest(BaseModel):
    type: Literal["openai_compatible", "anthropic", "openrouter"]
    base_url: str | None = None
    api_key: str
    model: str
    temperature: float | None = None


class StreamChatRequest(BaseModel):
    """Start or resume the user-facing routed chat."""

    thread_id: str | None = None
    conversation_id: str | None = None
    user_input: str | None = None
    resume_value: str | None = None
    router_provider: RuntimeProviderRequest | None = None


class UpdateProfileRequest(BaseModel):
    location: str | None = None
    timezone: str | None = None
    display_name: str | None = None
    personality: str | None = None
    router_provider: dict[str, Any] | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    router_api_key: str | None = None


class ToolEnabledRequest(BaseModel):
    enabled: bool


class ThreadResponse(BaseModel):
    thread_id: str


class StateSnapshotResponse(BaseModel):
    thread_id: str | None = None
    state: dict[str, Any] = Field(default_factory=dict)


class RunStatusResponse(BaseModel):
    thread_id: str
    run_id: str | None = None
    agent_key: str | None = None
    status: Literal["idle", "running", "interrupted", "completed", "error", "cancelled", "disconnected"]
    error: str | None = None
    updated_at: str | None = None


class ConversationMetadataResponse(BaseModel):
    conversation_id: str
    agent_key: str | None = None
    worked_seconds: int = 0
    last_thread_id: str | None = None
    last_run_id: str | None = None
    last_run_status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
