
import os

from dotenv import load_dotenv

from solidcue.providers.base import BaseProvider
from solidcue.providers.anthropic import AnthropicProvider
from solidcue.providers.openai_compatible import OpenAICompatibleProvider
from solidcue.providers.openrouter import OpenRouterProvider
from solidcue.observability import get_env_path

load_dotenv(dotenv_path=get_env_path())


def _build_provider(
    *,
    provider_type: str,
    api_key: str,
    model: str,
    base_url: str | None = None,
    temperature: float | None = None,
) -> BaseProvider:
    if provider_type == "openai_compatible":
        if not base_url:
            raise ValueError("base_url is required for openai_compatible provider")

        return OpenAICompatibleProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    if provider_type == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    if provider_type == "openrouter":
        return OpenRouterProvider(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider type: {provider_type}")


def get_api_key(api_key_env: str) -> str:
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise ValueError(f"Missing API key env var: {api_key_env}")

    return api_key


def get_provider(provider_config) -> BaseProvider:
    config = provider_config

    api_key = get_api_key(config.api_key_env)
    return _build_provider(
        provider_type=config.type,
        api_key=api_key,
        model=config.model,
        base_url=getattr(config, "base_url", None),
        temperature=getattr(config, "temperature", None),
    )


def get_provider_from_runtime_config(provider_config) -> BaseProvider:
    config = provider_config
    api_key = str(getattr(config, "api_key", "") or "").strip()
    if not api_key:
        raise ValueError("api_key is required for runtime provider")
    model = str(getattr(config, "model", "") or "").strip()
    if not model:
        raise ValueError("model is required for runtime provider")
    provider_type = str(getattr(config, "type", "") or "").strip()
    return _build_provider(
        provider_type=provider_type,
        api_key=api_key,
        model=model,
        base_url=getattr(config, "base_url", None),
        temperature=getattr(config, "temperature", None),
    )


def get_provider_from_any_config(provider_config) -> BaseProvider:
    config = provider_config
    api_key = str(getattr(config, "api_key", "") or "").strip()
    if not api_key:
        api_key_env = str(getattr(config, "api_key_env", "") or "").strip()
        if not api_key_env:
            raise ValueError("api_key or api_key_env is required for provider")
        api_key = get_api_key(api_key_env)
    model = str(getattr(config, "model", "") or "").strip()
    if not model:
        raise ValueError("model is required for provider")
    provider_type = str(getattr(config, "type", "") or "").strip()
    return _build_provider(
        provider_type=provider_type,
        api_key=api_key,
        model=model,
        base_url=getattr(config, "base_url", None),
        temperature=getattr(config, "temperature", None),
    )
