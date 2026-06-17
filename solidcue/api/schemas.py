"""HTTP request/response models for the API layer.

These models exist only to shape JSON at the transport boundary. Wherever a
service already defines an input model (e.g. ``CreateAgentInput``,
``CreateMcpToolInput``), the routes reuse it directly rather than redefining it
here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    last_thread_id: str | None = None
    last_run_id: str | None = None
    last_run_status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
