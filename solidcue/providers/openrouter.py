
from .openai_compatible import OpenAICompatibleProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: str, model: str):
        super().__init__(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            model=model,
        )

    def get_headers(self) -> dict:
        headers = super().get_headers()
        headers.update(
            {
                "HTTP-Referer": "http://localhost",
                "X-Title": "SolidCue",
            }
        )
        return headers
