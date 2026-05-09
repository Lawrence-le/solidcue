
import os

from dotenv import load_dotenv

from solidcue.providers.base import BaseProvider
from solidcue.providers.anthropic import AnthropicProvider
from solidcue.providers.openai_compatible import OpenAICompatibleProvider
from solidcue.providers.openrouter import OpenRouterProvider
from solidcue.utils.env import get_env_path

load_dotenv(dotenv_path=get_env_path())


def get_api_key(api_key_env: str) -> str:
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise ValueError(f"Missing API key env var: {api_key_env}")

    return api_key


def get_provider(provider_config) -> BaseProvider:
    config = provider_config

    api_key = get_api_key(config.api_key_env)

    if config.type == "openai_compatible":
        if not config.base_url:
            raise ValueError("base_url is required for openai_compatible provider")

        return OpenAICompatibleProvider(
            base_url=config.base_url,
            api_key=api_key,
            model=config.model,
        )

    if config.type == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=config.model,
        )

    if config.type == "openrouter":
        return OpenRouterProvider(
            api_key=api_key,
            model=config.model,
        )
    

    raise ValueError(f"Unsupported provider type: {config.type}")
