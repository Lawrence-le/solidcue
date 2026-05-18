from typing import Any

from .base import BaseProvider
from .client import HTTPClient


class OpenAICompatibleProvider(BaseProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float | None = None,
    ):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = HTTPClient()

    def _extract_usage(self, response: dict[str, Any]) -> dict[str, Any]:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return {}

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details")
        cached_tokens = 0
        if isinstance(prompt_details, dict):
            cached_tokens = int(prompt_details.get("cached_tokens") or 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "method": "provider_reported",
        }

    def get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _extract_text(self, response: dict) -> str:
        try:
            message = response["choices"][0]["message"]
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls")

            if tool_calls and not content.strip():
                call = tool_calls[0]["function"]
                name = call.get("name")
                args = call.get("arguments")
                return f"<|tool_call_start|>{name}(tool_input={args})<|tool_call_end|>"

            return content.strip()
        except (KeyError, IndexError, TypeError):
            return ""

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                continue

            item = dict(message)
            role = item.get("role")
            if role in {"system", "user", "assistant", "tool"}:
                content = item.get("content")
                if content is None:
                    item["content"] = ""
                elif not isinstance(content, str):
                    item["content"] = str(content)
            normalized.append(item)
        return normalized

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        normalized_messages = self._normalize_messages(messages)
        payload: dict[str, Any] = {"model": self.model, "messages": normalized_messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if isinstance(max_tokens, int) and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if isinstance(tools, list) and tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        data = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self.get_headers(),
            json=payload,
        )
        self.last_usage = self._extract_usage(data)
        output = self._extract_text(data)
        if not output and not data.get("choices"):
            raise ValueError(f"Empty/unsupported OpenAI-compatible response: {data}")
        return output
