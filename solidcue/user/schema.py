from typing import Any, Literal

from pydantic import BaseModel, Field


class RouterProviderConfig(BaseModel):
    type: Literal["openai_compatible", "anthropic", "openrouter"]
    base_url: str | None = None
    api_key_env: str
    model: str
    temperature: float | None = None


class UserProfileConfig(BaseModel):
    location: str | None = None
    timezone: str | None = None
    display_name: str | None = None
    personality: str | None = None
    router_provider: RouterProviderConfig | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
