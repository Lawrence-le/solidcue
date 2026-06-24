from __future__ import annotations

import json
from typing import Any

from solidcue.agent_configs.loader import list_agents
from solidcue.providers.factory import get_provider_from_any_config

_RUNTIME_ROUTER_PROVIDER_CONFIGS: dict[str, Any] = {}


def _load_profile_provider() -> Any:
    """Load the user's router provider at import time — never inside async nodes."""
    try:
        from solidcue.user.loader import load_user_profile

        cfg = load_user_profile().router_provider
        return get_provider_from_any_config(cfg) if cfg is not None else None
    except Exception:
        return None


# Loaded once so async nodes never block on file IO.
_PROFILE_ROUTER_PROVIDER: Any = _load_profile_provider()


def normalize_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def available_agents() -> list[dict[str, str]]:
    """The agents the router may route work to (key, name, description)."""
    agents: list[dict[str, str]] = []
    for agent in list_agents():
        agent_key = normalize_text(getattr(agent, "agent_key", ""))
        if not agent_key:
            continue
        agents.append(
            {
                "agent_key": agent_key,
                "name": normalize_text(getattr(agent, "name", "")),
                "description": normalize_text(getattr(agent, "description", "")),
            }
        )
    return agents


def resolve_router_provider(thread_id: str):
    """Runtime router provider for the thread, falling back to the profile provider.

    May raise ValueError if the configured provider is invalid (caller handles).
    """
    provider = get_runtime_router_provider(thread_id)
    if provider is None:
        provider = _PROFILE_ROUTER_PROVIDER
    return provider


def select_target_agent_key(user_input: str) -> str:
    lowered = user_input.casefold()
    available_agent_keys = {
        agent.agent_key for agent in list_agents() if isinstance(agent.agent_key, str)
    }

    if any(
        keyword in lowered for keyword in ("resume", "cv", "curriculum vitae", "cover letter")
    ):
        if "resume_builder" in available_agent_keys:
            return "resume_builder"

    if any(
        keyword in lowered
        for keyword in ("job description", "job posting", "job ad", "jd", "archive job")
    ):
        if "jd_archiver" in available_agent_keys:
            return "jd_archiver"

    if "resume_builder" in available_agent_keys:
        return "resume_builder"

    if available_agent_keys:
        return sorted(available_agent_keys)[0]

    return ""


def set_runtime_router_provider_config(thread_id: str, provider_config: Any) -> None:
    normalized_thread_id = normalize_text(thread_id)
    if not normalized_thread_id or provider_config is None:
        return
    _RUNTIME_ROUTER_PROVIDER_CONFIGS[normalized_thread_id] = provider_config


def get_runtime_router_provider(thread_id: str):
    normalized_thread_id = normalize_text(thread_id)
    if not normalized_thread_id:
        return None
    provider_config = _RUNTIME_ROUTER_PROVIDER_CONFIGS.get(normalized_thread_id)
    if provider_config is None:
        return None
    return get_provider_from_any_config(provider_config)


def clear_runtime_router_provider_config(thread_id: str) -> None:
    normalized_thread_id = normalize_text(thread_id)
    if not normalized_thread_id:
        return
    _RUNTIME_ROUTER_PROVIDER_CONFIGS.pop(normalized_thread_id, None)


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = normalize_text(text)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
