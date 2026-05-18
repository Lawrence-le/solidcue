from typing import Any

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
