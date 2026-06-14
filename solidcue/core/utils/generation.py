from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from solidcue.core.utils.metrics import build_metric, estimate_message_tokens, timed_generate

T = TypeVar("T")


async def stream_text_then_parse(
    provider: Any,
    messages: list[dict[str, Any]],
    parser: Callable[[str], T],
    *,
    stream_writer: Callable[[str], None] | None = None,
    node_name: str = "llm",
    max_tokens: int | None = None,
) -> tuple[T, dict[str, Any], str]:
    """Stream text from a provider, buffer it, then parse the final text."""
    estimated_stats = estimate_message_tokens(messages)
    model_name = str(getattr(provider, "model", "") or "").strip()
    started = time.perf_counter()
    chunks: list[str] = []

    try:
        async for chunk in provider.async_stream_generate(messages, max_tokens=max_tokens):
            if not isinstance(chunk, str) or not chunk:
                continue
            chunks.append(chunk)
            if callable(stream_writer):
                stream_writer(chunk)
    except Exception as exc:
        raise RuntimeError(f"{node_name} streaming failed: {exc}") from exc

    elapsed_s = time.perf_counter() - started
    output = "".join(chunks)

    provider_usage = {}
    get_last_usage = getattr(provider, "get_last_usage", None)
    if callable(get_last_usage):
        usage_payload = get_last_usage()
        if isinstance(usage_payload, dict):
            provider_usage = usage_payload

    output_tokens = 4 + max(1, len(output) // 4) if output else 0
    estimated_stats["estimated_assistant"] = output_tokens
    estimated_stats["estimated_total"] += output_tokens

    token_stats = dict(estimated_stats)
    if provider_usage:
        token_stats.update(
            {
                "prompt_tokens": int(provider_usage.get("prompt_tokens") or 0),
                "completion_tokens": int(provider_usage.get("completion_tokens") or 0),
                "total_tokens": int(provider_usage.get("total_tokens") or 0),
                "cached_tokens": int(provider_usage.get("cached_tokens") or 0),
                "method": str(provider_usage.get("method") or "provider_reported"),
            }
        )

    metric = build_metric(token_stats, elapsed_s, model_name)
    parsed = parser(output)
    return parsed, metric, output


def generate_full_then_parse(
    provider: Any,
    messages: list[dict[str, Any]],
    parser: Callable[[str], T],
    *,
    node_name: str = "llm",
    max_tokens: int | None = None,
) -> tuple[T, dict[str, Any], str]:
    """Generate a full response, then parse the final text."""
    output, metric = timed_generate(
        provider,
        messages,
        node_name=node_name,
        max_tokens=max_tokens,
    )
    parsed = parser(str(output or ""))
    return parsed, metric, str(output or "")
