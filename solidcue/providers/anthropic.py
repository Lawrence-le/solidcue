import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from .base import BaseProvider
from .client import HTTPClient

class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float | None = None,
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = HTTPClient()

    def _extract_usage(self, response: dict[str, Any]) -> dict[str, Any]:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return {}

        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
        cache_write_tokens = int(usage.get("cache_creation_input_tokens") or 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": cache_read_tokens + cache_write_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_input_tokens": cache_write_tokens,
            "method": "provider_reported",
        }

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url = "https://api.anthropic.com/v1/messages"

        prompt = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 1024,
            "messages": prompt,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        data = self.client.post(
            url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        self.last_usage = self._extract_usage(data)

        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected Anthropic response format: {data}"
            ) from e

    def _process_sse_event(self, event_name: str, data_lines: list[str]) -> list[str]:
        """Parse a complete SSE event, update usage state, and return any text chunks."""
        raw_payload = "\n".join(data_lines).strip()
        if not raw_payload:
            return []
        try:
            event_payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return []

        if event_name == "message_start":
            message = event_payload.get("message")
            if isinstance(message, dict):
                self.last_usage = self._extract_usage(message)
        elif event_name == "message_delta":
            usage = event_payload.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = int(self.last_usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("output_tokens") or 0)
                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cached_tokens": int(self.last_usage.get("cached_tokens") or 0),
                    "method": "provider_reported",
                }
        elif event_name == "content_block_delta":
            delta = event_payload.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    return [text]
        elif event_name == "error":
            error_payload = event_payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message")
                if isinstance(message, str) and message.strip():
                    raise RuntimeError(message.strip())
            raise RuntimeError(str(event_payload))
        return []

    def _build_stream_payload(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 1024,
            "messages": self._convert_messages(messages),
            "stream": True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def stream_generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = self._build_stream_payload(messages, max_tokens)
        self.last_usage = {}
        event_name = ""
        data_lines: list[str] = []

        for raw_line in self.client.stream_post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        ):
            line = raw_line.rstrip("\n")
            if not line:
                if data_lines:
                    yield from self._process_sse_event(event_name, data_lines)
                    data_lines = []
                    event_name = ""
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines:
            yield from self._process_sse_event(event_name, data_lines)

    async def async_stream_generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_stream_payload(messages, max_tokens)
        self.last_usage = {}
        event_name = ""
        data_lines: list[str] = []

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    line = raw_line.rstrip("\n")
                    if not line:
                        if data_lines:
                            for chunk in self._process_sse_event(event_name, data_lines):
                                yield chunk
                            data_lines = []
                            event_name = ""
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())

        if data_lines:
            for chunk in self._process_sse_event(event_name, data_lines):
                yield chunk

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """
        Convert OpenAI-style messages to Anthropic format
        """
        converted = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                converted.append(
                    {"role": "user", "content": f"[SYSTEM] {content}"}
                )
            else:
                converted.append({"role": role, "content": content})

        return converted
