from typing import Any

from pydantic import BaseModel, Field


class UserProfileConfig(BaseModel):
    location: str | None = None
    timezone: str | None = None
    display_name: str | None = None
    personality: str | None = None
    job_title: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
