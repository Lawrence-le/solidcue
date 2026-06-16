import os
from contextlib import contextmanager, nullcontext
from uuid import UUID
from typing import Any

from langgraph.config import get_config as _get_langgraph_config
from opentelemetry import trace as _otel_trace_api

from .common import is_truthy

_LANGFUSE_CLIENT: Any | None = None
_LANGFUSE_HANDLER: Any | None = None


class _LangfuseObservationHandle:
    __slots__ = ("attributes_context", "context", "observation")

    def __init__(
        self,
        *,
        context: Any,
        observation: Any,
        attributes_context: Any | None = None,
    ) -> None:
        self.attributes_context = attributes_context
        self.context = context
        self.observation = observation


def _normalize_trace_id(value: Any) -> str | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return UUID(raw).hex
    except ValueError:
        compact = raw.replace("-", "").lower()
        if len(compact) == 32 and all(c in "0123456789abcdef" for c in compact):
            return compact
    return None


def _langgraph_trace_context(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    if not metadata:
        return None
    trace_id = _normalize_trace_id(metadata.get("run_id"))
    return {"trace_id": trace_id} if trace_id else None


def _default_langgraph_trace_name(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None

    configured = metadata.get("langfuse_trace_name")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    graph_id = metadata.get("graph_id")
    if isinstance(graph_id, str) and graph_id.strip():
        return f"solidcue:{graph_id.strip()}"

    return None


def _langgraph_trace_attributes(
    metadata: dict[str, Any] | None,
) -> dict[str, str | dict[str, str]] | None:
    if not metadata:
        return None

    attributes: dict[str, str | dict[str, str]] = {}
    trace_context = _langgraph_trace_context(metadata)
    if trace_context is not None:
        attributes["trace_context"] = trace_context

    trace_name = _default_langgraph_trace_name(metadata)
    if trace_name:
        attributes["trace_name"] = trace_name

    session_id = metadata.get("langfuse_session_id") or metadata.get("thread_id")
    if isinstance(session_id, str) and session_id.strip():
        attributes["session_id"] = session_id.strip()

    return attributes or None


def _current_langgraph_trace_attributes() -> dict[str, str | dict[str, str]] | None:
    try:
        config = _get_langgraph_config()
    except Exception:
        return None

    metadata = config.get("metadata") if isinstance(config, dict) else None
    return _langgraph_trace_attributes(metadata if isinstance(metadata, dict) else None)


def _has_current_otel_span() -> bool:
    try:
        return _otel_trace_api.get_current_span().get_span_context().is_valid
    except Exception:
        return False


def is_langfuse_enabled() -> bool:
    return is_truthy(os.getenv("SOLIDCUE_LANGFUSE_ENABLED"))


def _bootstrap_langfuse() -> None:
    global _LANGFUSE_CLIENT, _LANGFUSE_HANDLER

    if _LANGFUSE_CLIENT is not None and _LANGFUSE_HANDLER is not None:
        return

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler as _Base
    except ImportError:
        return

    try:
        class _Handler(_Base):
            """Bridges LangGraph Server's thread_id → langfuse_session_id.

            LangGraph Server injects thread_id into run metadata at runtime.
            Langfuse reads langfuse_session_id from that same metadata dict.
            This subclass adds the key when it's missing so all turns of a
            conversation are grouped under one Langfuse session automatically.
            """

            def _parse_langfuse_trace_attributes(self, *, metadata, tags):
                if metadata:
                    metadata = dict(metadata)
                    if not metadata.get("langfuse_session_id") and metadata.get("thread_id"):
                        metadata["langfuse_session_id"] = metadata["thread_id"]
                    if not metadata.get("langfuse_trace_name"):
                        trace_name = _default_langgraph_trace_name(metadata)
                        if trace_name:
                            metadata["langfuse_trace_name"] = trace_name
                return super()._parse_langfuse_trace_attributes(
                    metadata=metadata, tags=tags
                )

            def _take_root_trace_context(self, *, inputs, metadata):
                if _has_current_otel_span():
                    return None, None

                trace_context = _langgraph_trace_context(metadata)
                if trace_context is not None:
                    return None, trace_context
                return super()._take_root_trace_context(
                    inputs=inputs,
                    metadata=metadata,
                )

        _LANGFUSE_CLIENT = get_client()
        _LANGFUSE_HANDLER = _Handler()
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


@contextmanager
def start_langfuse_span(
    *,
    name: str,
    input_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    if not is_langfuse_enabled():
        with nullcontext():
            yield None
        return

    _bootstrap_langfuse()
    if _LANGFUSE_CLIENT is None:
        with nullcontext():
            yield None
        return

    observation_kwargs: dict[str, Any] = {
        "name": name,
        "as_type": "span",
        "input": input_payload,
        "metadata": metadata or None,
    }

    # When there is no active OTEL span (e.g. a parallel LangGraph branch that
    # lost the parent context), attach to the run's trace via the run_id-derived
    # trace_context + session, so the span nests under the same trace instead of
    # starting a new root one. Mirrors start_langfuse_generation.
    attributes_context = None
    try:
        trace_attributes = None
        try:
            if _LANGFUSE_CLIENT.get_current_trace_id() is None:
                trace_attributes = _current_langgraph_trace_attributes()
        except Exception:
            trace_attributes = None

        if trace_attributes is not None:
            trace_context = trace_attributes.get("trace_context")
            if isinstance(trace_context, dict):
                observation_kwargs["trace_context"] = trace_context

            trace_name = trace_attributes.get("trace_name")
            session_id = trace_attributes.get("session_id")
            if isinstance(trace_name, str) or isinstance(session_id, str):
                from langfuse import propagate_attributes

                attributes_context = propagate_attributes(
                    trace_name=trace_name if isinstance(trace_name, str) else None,
                    session_id=session_id if isinstance(session_id, str) else None,
                )
                attributes_context.__enter__()

        context = _LANGFUSE_CLIENT.start_as_current_observation(**observation_kwargs)
    except Exception:
        if attributes_context is not None:
            try:
                attributes_context.__exit__(None, None, None)
            except Exception:
                pass
        with nullcontext():
            yield None
        return

    try:
        with context as span:
            yield span
    finally:
        if attributes_context is not None:
            try:
                attributes_context.__exit__(None, None, None)
            except Exception:
                pass


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
        trace_attributes = None
        try:
            if _LANGFUSE_CLIENT.get_current_trace_id() is None:
                trace_attributes = _current_langgraph_trace_attributes()
        except Exception:
            trace_attributes = None

        observation_kwargs = {
            "name": name,
            "as_type": "generation",
            "input": input_payload,
            "model": model or None,
            "model_parameters": model_parameters or None,
            "end_on_exit": False,
        }

        attributes_context = None
        if trace_attributes is not None:
            trace_context = trace_attributes.get("trace_context")
            if isinstance(trace_context, dict):
                observation_kwargs["trace_context"] = trace_context

            trace_name = trace_attributes.get("trace_name")
            session_id = trace_attributes.get("session_id")
            if isinstance(trace_name, str) or isinstance(session_id, str):
                from langfuse import propagate_attributes

                attributes_context = propagate_attributes(
                    trace_name=trace_name if isinstance(trace_name, str) else None,
                    session_id=session_id if isinstance(session_id, str) else None,
                )
                attributes_context.__enter__()

        context = _LANGFUSE_CLIENT.start_as_current_observation(**observation_kwargs)
        return _LangfuseObservationHandle(
            attributes_context=attributes_context,
            context=context,
            observation=context.__enter__(),
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

    context = None
    attributes_context = None
    observation = generation
    if isinstance(generation, _LangfuseObservationHandle):
        attributes_context = generation.attributes_context
        context = generation.context
        observation = generation.observation

    try:
        observation.update(
            output=output_payload,
            usage_details=usage_details or None,
            metadata=metadata or None,
        )
        observation.end()
    except Exception:
        pass
    finally:
        if context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception:
                return
        if attributes_context is not None:
            try:
                attributes_context.__exit__(None, None, None)
            except Exception:
                return
