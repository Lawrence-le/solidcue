"""User-profile endpoints — wraps the existing ``solidcue.user.loader``.

CLI parity: `setup-view` (GET) and `setup-init` / `setup-update` (PUT). Purely
additive — reuses `load_user_profile` / `save_user_profile`, no existing file changed.
"""

from __future__ import annotations

from fastapi import APIRouter

from solidcue.api.schemas import UpdateProfileRequest
from solidcue.observability import upsert_env_key
from solidcue.user.loader import load_user_profile, save_user_profile
from solidcue.user.schema import UserProfileConfig

router = APIRouter(prefix="/profile", tags=["profile"])


def _router_env_key(provider_type: str) -> str:
    normalized = str(provider_type or "").strip()
    mapping = {
        "openai_compatible": "ROUTER_OPENAI_COMPATIBLE_API_KEY",
        "anthropic": "ROUTER_ANTHROPIC_API_KEY",
        "openrouter": "ROUTER_OPENROUTER_API_KEY",
    }
    env_key = mapping.get(normalized)
    if not env_key:
        raise ValueError(f"Unsupported router provider type: {normalized}")
    return env_key


@router.get("", response_model=UserProfileConfig)
def get_profile() -> UserProfileConfig:
    return load_user_profile()


@router.put("", response_model=UserProfileConfig)
def update_profile(profile: UpdateProfileRequest) -> UserProfileConfig:
    data = profile.model_dump()
    router_provider = data.get("router_provider")
    router_api_key = str(data.pop("router_api_key", "") or "").strip()
    if isinstance(router_provider, dict):
        router_provider["api_key_env"] = _router_env_key(router_provider.get("type"))
        if router_api_key:
            upsert_env_key(router_provider["api_key_env"], router_api_key)
    profile_config = UserProfileConfig(**data)
    if isinstance(router_provider, dict):
        profile_config.router_provider = profile_config.router_provider.model_copy(
            update={"api_key_env": router_provider["api_key_env"]}
        ) if profile_config.router_provider is not None else None
    save_user_profile(profile_config)
    return load_user_profile()
