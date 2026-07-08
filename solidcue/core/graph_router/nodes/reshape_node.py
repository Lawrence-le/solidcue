"""
reshape_node: answers a follow-up that only re-presents already-gathered data.

When the user asks to reformat / add a column / change units on data an agent
already returned, there is no need to re-dispatch the agent. This node re-synthesises
a response from the structured ``data`` retained on prior ``agent_results`` entries,
so the same snapshot is reused (no extra tool calls, no timestamp drift).
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer

from solidcue.core.graph_router.nodes._shared import (
    _PROFILE_ROUTER_PROVIDER,
    normalize_text,
)
from solidcue.core.graph_router.prompts.router_reshape_prompt import (
    build_router_reshape_messages,
)
from solidcue.core.graph_router.state.schema import RouterState

logger = logging.getLogger(__name__)


def _has_reusable_data(agent_results: list[dict[str, Any]] | None) -> bool:
    if not isinstance(agent_results, list):
        return False
    return any(
        isinstance(r, dict) and isinstance(r.get("data"), dict) and r.get("data")
        for r in agent_results
    )


def _has_reusable_history(chat_history: list[dict[str, Any]] | None) -> bool:
    """True when a prior turn already produced content we can re-render.

    Data answered directly by the `chat` intent never lands in ``agent_results``
    (no agent ran); it lives only in CHAT_HISTORY. A reshape of that content is
    still valid, so we treat any prior assistant message as a reusable source.
    """
    if not isinstance(chat_history, list):
        return False
    return any(
        isinstance(m, dict)
        and str(m.get("role") or "").strip() == "assistant"
        and str(m.get("content") or "").strip()
        for m in chat_history
    )


async def reshape_node(state: RouterState) -> dict[str, Any]:
    writer = get_stream_writer()
    user_input = normalize_text(state.get("user_input"))
    agent_results: list[dict[str, Any]] = list(state.get("agent_results") or [])
    chat_history = state.get("chat_history")

    # Reshape can re-render structured agent data OR content a prior `chat` turn
    # already produced (which lives only in CHAT_HISTORY, never in agent_results).
    # Only bail when there is neither — an honest message rather than fabricating.
    if not _has_reusable_data(agent_results) and not _has_reusable_history(chat_history):
        message = (
            "I don't have the earlier results saved to reshape. "
            "Ask me to fetch the data again and I'll get fresh values."
        )
        writer({"event": "message_delta", "data": {"delta": message}})
        return {"final_response": message, "synthesis_draft": message}

    provider = _PROFILE_ROUTER_PROVIDER
    synthesis = ""

    if provider is not None:
        messages = build_router_reshape_messages(
            user_input=user_input,
            agent_results=agent_results,
            chat_history=chat_history,
        )
        chunks: list[str] = []
        try:
            async for chunk in provider.async_stream_generate(messages):
                if chunk:
                    chunks.append(chunk)
                    writer({"event": "message_delta", "data": {"delta": chunk}})
        except Exception:
            logger.exception("reshape synthesis streaming failed")
            chunks = []
        synthesis = "".join(chunks)

    if not synthesis:
        # No provider or generation failed — reuse the most recent rendered output so
        # the user at least sees the prior result rather than nothing.
        for result in reversed(agent_results):
            output = normalize_text(result.get("output")) if isinstance(result, dict) else ""
            if output:
                synthesis = output
                break
        if not synthesis and isinstance(chat_history, list):
            for message in reversed(chat_history):
                if not isinstance(message, dict):
                    continue
                if str(message.get("role") or "").strip() != "assistant":
                    continue
                content = normalize_text(message.get("content"))
                if content:
                    synthesis = content
                    break
        synthesis = synthesis or normalize_text(state.get("assistant_draft")) or "Done."
        writer({"event": "message_delta", "data": {"delta": synthesis}})

    return {"final_response": synthesis, "synthesis_draft": synthesis}
