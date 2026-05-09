import os
from threading import Lock
from typing import Any

_TRACING_BOOTSTRAPPED = False
_TRACING_LOCK = Lock()


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_primitive(value: Any) -> bool | int | float | str:
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def is_phoenix_enabled() -> bool:
    explicit = os.getenv("SOLIDCUE_PHOENIX_ENABLED")
    if explicit is not None:
        return _is_truthy(explicit)
    return _is_truthy(os.getenv("PHOENIX_ENABLED"))


def is_langsmith_enabled() -> bool:
    explicit = os.getenv("SOLIDCUE_LANGSMITH_ENABLED")
    if explicit is not None:
        return _is_truthy(explicit)
    return _is_truthy(os.getenv("LANGSMITH_TRACING"))


def configure_langsmith_tracing_env() -> None:
    # Ensure runtime behavior follows SolidCue-level toggle even when LANGSMITH_TRACING
    # is inherited from parent shell/profile.
    os.environ["LANGSMITH_TRACING"] = "true" if is_langsmith_enabled() else "false"


def ensure_phoenix_tracing() -> None:
    global _TRACING_BOOTSTRAPPED
    if _TRACING_BOOTSTRAPPED or not is_phoenix_enabled():
        return

    with _TRACING_LOCK:
        if _TRACING_BOOTSTRAPPED:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            return

        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
        service_name = os.getenv("PHOENIX_SERVICE_NAME", "solidcue")

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=endpoint,
                )
            )
        )
        trace.set_tracer_provider(provider)
        _TRACING_BOOTSTRAPPED = True


def trace_langgraph_invoke(
    *,
    span_name: str,
    attributes: dict[str, Any],
    invoke: Any,
) -> Any:
    if not is_phoenix_enabled():
        return invoke()

    ensure_phoenix_tracing()
    try:
        from opentelemetry import trace
    except ImportError:
        return invoke()

    tracer = trace.get_tracer("solidcue.langgraph")
    with tracer.start_as_current_span(span_name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, _to_primitive(value))
        return invoke()
