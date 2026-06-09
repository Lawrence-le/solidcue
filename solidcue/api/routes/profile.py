"""User-profile endpoints — wraps the existing ``solidcue.user.loader``.

CLI parity: `setup-view` (GET) and `setup-init` / `setup-update` (PUT). Purely
additive — reuses `load_user_profile` / `save_user_profile`, no existing file changed.
"""

from __future__ import annotations

from fastapi import APIRouter

from solidcue.user.loader import load_user_profile, save_user_profile
from solidcue.user.schema import UserProfileConfig

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("", response_model=UserProfileConfig)
def get_profile() -> UserProfileConfig:
    return load_user_profile()


@router.put("", response_model=UserProfileConfig)
def update_profile(profile: UserProfileConfig) -> UserProfileConfig:
    save_user_profile(profile)
    return load_user_profile()
