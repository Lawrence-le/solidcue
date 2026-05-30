from typing import Any


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def to_primitive(value: Any) -> bool | int | float | str:
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
