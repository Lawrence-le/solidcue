from pathlib import Path

import yaml

from solidcue.user.schema import UserProfileConfig


USER_CONFIG_DIR = Path(__file__).parent / "configs"
USER_PROFILE_PATH = USER_CONFIG_DIR / "user_profile.yaml"


def get_user_profile_path() -> Path:
    return USER_PROFILE_PATH


def load_user_profile() -> UserProfileConfig:
    if not USER_PROFILE_PATH.exists():
        return UserProfileConfig()

    with USER_PROFILE_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    return UserProfileConfig(**data)


def save_user_profile(profile: UserProfileConfig) -> Path:
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    with USER_PROFILE_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(profile.model_dump(exclude_none=True), file, sort_keys=False)

    return USER_PROFILE_PATH
