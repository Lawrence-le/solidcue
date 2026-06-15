"""
execute_plan_node: runs the multi-agent task plan inside the router graph.

Each step in state["plan"] runs the appropriate agent subgraph and collects its
final_response. Custom events dispatched via get_stream_writer() propagate to the
parent graph's "custom" stream mode — the frontend receives them via the same
join_stream that covers the whole task run.

Ported from run_engine.py:1620-1800. Queue/SSE/side-DB logic is dropped; replaced
by graph-native streaming via the outer get_stream_writer().
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_config as _get_config, get_stream_writer

from solidcue.core.graph_router.nodes._shared import (
    _PROFILE_ROUTER_PROVIDER,
    normalize_text,
)
from solidcue.core.graph_router.prompts.router_synthesis_prompt import (
    build_router_synthesis_messages,
)
from solidcue.core.graph_router.state.schema import RouterState
from solidcue.observability.langfuse import start_langfuse_span

logger = logging.getLogger(__name__)

# Cache compiled agent graphs by agent_key so we don't rebuild on every step.
_agent_graph_cache: dict[str, Any] = {}


def _get_agent_graph(agent_key: str) -> Any:
    if agent_key not in _agent_graph_cache:
        from solidcue.core.graph_agent.builder import _compile_graph

        _agent_graph_cache[agent_key] = _compile_graph(
            checkpointer=None,
            include_langfuse_callbacks=False,
        )
    return _agent_graph_cache[agent_key]


def _compose_subtask_input(
    *,
    sub_task: str,
    user_input: str,
    prior_results: list[dict[str, Any]],
) -> str:
    """Port of run_engine._compose_subtask_input (line 542)."""
    base = (sub_task or user_input).strip()
    context_blocks: list[str] = []
    for result in prior_results:
        output = str(result.get("output") or "").strip()
        if not output:
            continue
        agent_key = str(result.get("agent_key") or "agent").strip()
        context_blocks.append(f"From {agent_key}:\n{output}")
    if not context_blocks:
        return base
    return base + "\n\nContext from earlier steps:\n" + "\n\n".join(context_blocks)


def _extract_message_delta(chunk: Any) -> str:
    """Extract text delta from a messages-mode chunk (list of [AIMessageChunk, meta])."""
    msg = chunk[0] if isinstance(chunk, (list, tuple)) else chunk
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    return text
    return ""


async def execute_plan_node(state: RouterState) -> dict[str, Any]:
    writer = get_stream_writer()
    plan: list[dict[str, Any]] = list(state.get("plan") or [])
    user_input = normalize_text(state.get("user_input"))

    if not plan:
        return {"synthesis_draft": "", "agent_results": []}

    step_count = len(plan)

    # 1. Announce the full plan so the UI can render step cards immediately.
    writer({
        "event": "plan",
        "data": {
            "intro": normalize_text(state.get("assistant_draft")),
            "route_reason": normalize_text(state.get("route_reason")),
            "steps": [
                {
                    "agent_key": step.get("agent_key", ""),
                    "sub_task": step.get("sub_task", ""),
                    "step_index": idx,
                }
                for idx, step in enumerate(plan)
            ],
            "step_count": step_count,
        },
    })

    # 2. Run each step sequentially, forwarding token deltas as they arrive.
    #
    _parent_config = _get_config()
    _parent_metadata: dict[str, Any] = dict(_parent_config.get("metadata") or {})
    _parent_callbacks = _parent_config.get("callbacks")

    agent_results: list[dict[str, Any]] = []

    for step_index, step in enumerate(plan):
        agent_key = normalize_text(step.get("agent_key"))
        sub_task = normalize_text(step.get("sub_task")) or user_input

        writer({
            "event": "subagent",
            "data": {
                "agent_key": agent_key,
                "sub_task": sub_task,
                "step_index": step_index,
                "step_count": step_count,
                "status": "running",
            },
        })

        composed_input = _compose_subtask_input(
            sub_task=sub_task,
            user_input=user_input,
            prior_results=agent_results,
        )

        captured_output = ""
        step_status = "completed"

        try:
            agent_graph = _get_agent_graph(agent_key)
            agent_input: dict[str, Any] = {
                "user_input": composed_input,
                "agent_key": agent_key,
                "chat_history": list(state.get("chat_history") or []),
            }
            config: dict[str, Any] = {
                "configurable": {"agent_key": agent_key},
                "metadata": _parent_metadata,
            }
            if _parent_callbacks is not None:
                config["callbacks"] = _parent_callbacks

            with start_langfuse_span(
                name=f"solidcue:agent:{agent_key}",
                input_payload={"sub_task": sub_task, "step_index": step_index},
                metadata={
                    "agent_key": agent_key,
                    "step_index": step_index,
                    "step_count": step_count,
                },
            ):
                async for mode, chunk in agent_graph.astream(
                    agent_input,
                    config=config,
                    stream_mode=["updates", "messages"],
                ):
                    if mode == "messages":
                        delta = _extract_message_delta(chunk)
                        if delta:
                            writer({
                                "event": "subagent_delta",
                                "data": {
                                    "agent_key": agent_key,
                                    "step_index": step_index,
                                    "delta": delta,
                                },
                            })

                    elif mode == "updates" and isinstance(chunk, dict):
                        final_update = chunk.get("final_output")
                        if isinstance(final_update, dict):
                            response = normalize_text(final_update.get("final_response"))
                            if response:
                                captured_output = response

        except Exception:
            logger.exception("execute_plan step %d (%s) failed", step_index, agent_key)
            step_status = "failed"
            captured_output = f"Step {step_index + 1} failed."

        agent_results.append({
            "agent_key": agent_key,
            "sub_task": sub_task,
            "output": captured_output,
            "status": step_status,
        })

        writer({
            "event": "subagent",
            "data": {
                "agent_key": agent_key,
                "sub_task": sub_task,
                "step_index": step_index,
                "step_count": step_count,
                "status": step_status,
                "output": captured_output,
            },
        })

    # 3. Synthesise a unified response from all step outputs.
    synthesis = ""
    provider = _PROFILE_ROUTER_PROVIDER

    if provider is not None:
        synth_messages = build_router_synthesis_messages(
            user_input=user_input,
            agent_results=agent_results,
            chat_history=state.get("chat_history"),
        )
        synth_chunks: list[str] = []
        try:
            async for chunk in provider.async_stream_generate(synth_messages):
                if chunk:
                    synth_chunks.append(chunk)
                    writer({"event": "message_delta", "data": {"delta": chunk}})
        except Exception:
            logger.exception("execute_plan synthesis streaming failed")
            synth_chunks = []
        synthesis = "".join(synth_chunks)

    if not synthesis:
        # Fallback: stitch step outputs directly without an LLM call.
        parts = [
            str(r.get("output") or "").strip()
            for r in agent_results
            if str(r.get("output") or "").strip()
        ]
        synthesis = "\n\n".join(parts) or normalize_text(state.get("assistant_draft")) or "Done."
        writer({"event": "message_delta", "data": {"delta": synthesis}})

    return {
        "synthesis_draft": synthesis,
        "final_response": synthesis,
        "agent_results": agent_results,
    }
