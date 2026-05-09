import os
from pathlib import Path

from dotenv import set_key


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_env_path() -> Path:
    configured_path = Path(
        os.environ.get("SOLIDCUE_ENV_PATH", str(get_project_root() / ".env"))
    )
    return configured_path if configured_path.is_absolute() else (Path.cwd() / configured_path)


def generate_env_key(agent_key: str) -> str:
    return f"{agent_key.upper()}_API_KEY"


def write_env_key(env_key: str, value: str):
    normalized = value.strip()
    if not normalized:
        raise ValueError("API key cannot be empty")

    env_path = get_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)

    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")

        if f"{env_key}=" in content:
            raise ValueError(f"{env_key} already exists in .env")
    else:
        env_path.touch()

    set_key(str(env_path), env_key, normalized, quote_mode="never")
