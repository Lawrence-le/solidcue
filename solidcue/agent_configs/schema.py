from pydantic import BaseModel, Field
from typing import Any, Literal
from uuid import uuid4


class ProviderConfig(BaseModel):
    type: str
    base_url: str | None = None
    api_key_env: str
    model: str
    temperature: float | None = None


class PlanningPolicy(BaseModel):
    # static  -> deterministic pipeline; cache the task plan once and reuse it.
    # dynamic -> plan shape varies per request; re-plan every turn, never cache.
    # Default is dynamic: a wrong cached plan is more damaging than re-planning.
    mode: Literal["static", "dynamic"] = "dynamic"


class AgentConfig(BaseModel):
    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_key: str
    name: str
    description: str = ""

    provider: ProviderConfig
    lite_provider: ProviderConfig | None = None
    reviewer_provider: ProviderConfig | None = None
    writer_provider: ProviderConfig | None = None

    tools: list[str] = Field(default_factory=list)
    allowed_tasks: list[str] = Field(default_factory=list)

    style: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)

    planning: PlanningPolicy = Field(default_factory=PlanningPolicy)
    validation_policy: str | None = None
