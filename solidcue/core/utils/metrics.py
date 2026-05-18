from __future__ import annotations

import math
import time
from typing import Any


_CHAT_MESSAGE_OVERHEAD = 4
_CHAT_REPLY_PRIMING = 2


def estimate_text_tokens(text: str) -> int:
    value = text or ""
    if not value:
        return 0
    return max(1, math.ceil(len(value) / 4))


def estimate_message_tokens(messages: list[dict[str, Any]]) -> dict[str, Any]:
    system_tokens = 0
    user_tokens = 0
    assistant_tokens = 0
    tool_tokens = 0
    other_tokens = 0
    total_tokens = _CHAT_REPLY_PRIMING

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "other")
        content = message.get("content")
        content_text = content if isinstance(content, str) else str(content or "")
        message_tokens = _CHAT_MESSAGE_OVERHEAD + estimate_text_tokens(content_text)
        total_tokens += message_tokens

        if role == "system":
            system_tokens += message_tokens
        elif role == "user":
            user_tokens += message_tokens
        elif role == "assistant":
            assistant_tokens += message_tokens
        elif role == "tool":
            tool_tokens += message_tokens
        else:
            other_tokens += message_tokens

    return {
        "estimated_total": total_tokens,
        "estimated_system": system_tokens,
        "estimated_user": user_tokens,
        "estimated_assistant": assistant_tokens,
        "estimated_tool": tool_tokens,
        "estimated_other": other_tokens,
        "message_count": len(messages),
        "method": "approx_chars_div_4_plus_chat_overhead",
    }


def _provider_model(provider: Any) -> str:
    model = getattr(provider, "model", None)
    return str(model).strip() if isinstance(model, str) and str(model).strip() else ""


def build_metric(tokens: dict[str, Any] | None, elapsed_s: float = 0.0, model: str = "") -> dict[str, Any]:
    stats = tokens if isinstance(tokens, dict) else {}
    return {
        "tokens": stats,
        "time_s": max(0.0, float(elapsed_s)),
        "model": str(model or "").strip(),
    }


def build_metric_usage_event(node: str, metric: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metric, dict) or not metric:
        return None
    stats = metric.get("tokens")
    if not isinstance(stats, dict) or not stats:
        return None
    return {
        "node": node,
        "model": str(metric.get("model") or "").strip(),
        "prompt_tokens": int(stats.get("prompt_tokens") or 0),
        "completion_tokens": int(stats.get("completion_tokens") or 0),
        "total_tokens": int(stats.get("total_tokens") or 0),
        "cached_tokens": int(stats.get("cached_tokens") or 0),
        "estimated_total": int(stats.get("estimated_total") or 0),
        "estimated_system": int(stats.get("estimated_system") or 0),
        "estimated_user": int(stats.get("estimated_user") or 0),
        "estimated_assistant": int(stats.get("estimated_assistant") or 0),
        "estimated_tool": int(stats.get("estimated_tool") or 0),
        "estimated_other": int(stats.get("estimated_other") or 0),
        "message_count": int(stats.get("message_count") or 0),
        "llm_call_count": int(
            stats.get("llm_call_count")
            or (
                1
                if int(stats.get("total_tokens") or 0) > 0
                or int(stats.get("estimated_total") or 0) > 0
                else 0
            )
        ),
        "time_s": float(metric.get("time_s") or 0.0),
        "method": str(stats.get("method") or "approx_chars_div_4_plus_chat_overhead"),
    }


def timed_generate(
    provider: Any,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    estimated_stats = estimate_message_tokens(messages)
    started = time.perf_counter()
    output = provider.generate(messages, max_tokens=max_tokens)
    elapsed_s = time.perf_counter() - started

    output_tokens = _CHAT_MESSAGE_OVERHEAD + estimate_text_tokens(output or "")
    estimated_stats["estimated_assistant"] = output_tokens
    estimated_stats["estimated_total"] += output_tokens

    provider_usage = {}
    get_last_usage = getattr(provider, "get_last_usage", None)
    if callable(get_last_usage):
        usage_payload = get_last_usage()
        if isinstance(usage_payload, dict):
            provider_usage = usage_payload

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

    return output, build_metric(token_stats, elapsed_s, _provider_model(provider))


def build_metric_state_delta(node: str, metric_key: str, metric: dict[str, Any] | None) -> dict[str, Any]:
    event = build_metric_usage_event(node, metric)
    return {
        metric_key: metric or {},
        "metric_usage_events": [event] if event else [],
    }
