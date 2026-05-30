import os

from .common import is_truthy


def is_langsmith_enabled() -> bool:
    explicit = os.getenv("SOLIDCUE_LANGSMITH_ENABLED")
    if explicit is not None:
        return is_truthy(explicit)
    return is_truthy(os.getenv("LANGSMITH_TRACING"))


def configure_langsmith_tracing_env() -> None:
    # Keep runtime behavior aligned with SolidCue toggle, even if parent shell
    # exports LANGSMITH_TRACING.
    os.environ["LANGSMITH_TRACING"] = "true" if is_langsmith_enabled() else "false"
