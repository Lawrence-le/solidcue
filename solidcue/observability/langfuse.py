from contextlib import contextmanager, nullcontext
import os
from typing import Any

from .common import is_truthy

_LANGFUSE_CLIENT: Any | None = None
_LANGFUSE_HANDLER: Any | None = None


def is_langfuse_enabled() -> bool:
    return is_truthy(os.getenv("SOLIDCUE_LANGFUSE_ENABLED"))


def _bootstrap_langfuse() -> None:
    global _LANGFUSE_CLIENT, _LANGFUSE_HANDLER

    if _LANGFUSE_CLIENT is not None and _LANGFUSE_HANDLER is not None:
        return

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return

    try:
        _LANGFUSE_CLIENT = get_client()
        _LANGFUSE_HANDLER = CallbackHandler()
    except Exception:
        # Fail-open: observability must not break execution.
        _LANGFUSE_CLIENT = None
        _LANGFUSE_HANDLER = None


def get_langfuse_callbacks() -> list[Any]:
    if not is_langfuse_enabled():
        return []

    _bootstrap_langfuse()
    if _LANGFUSE_HANDLER is None:
        return []
    return [_LANGFUSE_HANDLER]


def flush_langfuse() -> None:
    if _LANGFUSE_CLIENT is None:
        return
    try:
        _LANGFUSE_CLIENT.flush()
    except Exception:
        # Fail-open for short-lived CLI use.
        return


def _is_valid_langfuse_session_id(value: str) -> bool:
    if not value or len(value) > 200:
        return False
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


@contextmanager
def propagate_langfuse_session(*, session_id: str | None):
    if not is_langfuse_enabled():
        with nullcontext():
            yield
        return

    normalized = str(session_id or "").strip()
    if not _is_valid_langfuse_session_id(normalized):
        with nullcontext():
            yield
        return

    try:
        from langfuse import propagate_attributes
    except ImportError:
        with nullcontext():
            yield
        return
    except Exception:
        with nullcontext():
            yield
        return

    try:
        context = propagate_attributes(session_id=normalized)
    except Exception:
        with nullcontext():
            yield
        return
    with context:
        yield


@contextmanager
def start_langfuse_root_span(*, name: str, input_payload: Any | None = None):
    if not is_langfuse_enabled():
        with nullcontext():
            yield
        return

    _bootstrap_langfuse()
    if _LANGFUSE_CLIENT is None:
        with nullcontext():
            yield
        return

    try:
        context = _LANGFUSE_CLIENT.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_payload,
        )
    except Exception:
        with nullcontext():
            yield
        return
    with context:
        yield


def start_langfuse_generation(
    *,
    name: str,
    input_payload: Any,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> Any | None:
    if not is_langfuse_enabled():
        return None

    _bootstrap_langfuse()
    if _LANGFUSE_CLIENT is None:
        return None

    try:
        return _LANGFUSE_CLIENT.start_observation(
            name=name,
            as_type="generation",
            input=input_payload,
            model=model or None,
            model_parameters=model_parameters or None,
        )
    except Exception:
        return None


def end_langfuse_generation(
    generation: Any | None,
    *,
    output_payload: Any,
    usage_details: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if generation is None:
        return

    try:
        generation.update(
            output=output_payload,
            usage_details=usage_details or None,
            metadata=metadata or None,
        )
        generation.end()
    except Exception:
        return
