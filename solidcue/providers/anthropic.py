from .base import BaseProvider
from .client import HTTPClient

class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = HTTPClient()

    def generate(self, messages: list[dict]) -> str:
        url = "https://api.anthropic.com/v1/messages"

        prompt = self._convert_messages(messages)

        data = self.client.post(
            url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1024,
                "messages": prompt,
            },
        )

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
