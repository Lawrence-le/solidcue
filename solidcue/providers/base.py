
import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
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

    def stream_generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream chat content in chunks.

        Providers without native streaming support fall back to a single
        blocking `generate()` call so streaming callers can keep one code path.
        """
        output = self.generate(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        if output:
            yield output

    async def async_stream_generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Async stream chat content in chunks.

        Default falls back to sync `generate()` for providers without native
        async streaming — subclasses should override with a true async implementation.
        The blocking call is offloaded to a worker thread so it never freezes the
        event loop when awaited from an async graph node.
        """
        output = await asyncio.to_thread(
            self.generate,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        if output:
            yield output
