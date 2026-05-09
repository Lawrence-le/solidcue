from .base import BaseProvider
from .client import HTTPClient


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = HTTPClient()

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

    def generate(self, messages: list[dict]) -> str:
        normalized_messages = self._normalize_messages(messages)
        data = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self.get_headers(),
            json={"model": self.model, "messages": normalized_messages},
        )
        output = self._extract_text(data)
        if not output and not data.get("choices"):
            raise ValueError(f"Empty/unsupported OpenAI-compatible response: {data}")
        return output
