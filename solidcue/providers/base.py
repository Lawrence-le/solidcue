
from abc import ABC, abstractmethod
from typing import Any


class BaseProvider(ABC):
    def __init__(self) -> None:
        self.last_usage: dict[str, Any] = {}

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat messages to LLM and return normalized response."""
        pass

    def get_last_usage(self) -> dict[str, Any]:
        usage = getattr(self, "last_usage", {})
        return usage if isinstance(usage, dict) else {}
