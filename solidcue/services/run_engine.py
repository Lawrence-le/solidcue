"""Execution engine for LangGraph agent runs.

Owns the full run lifecycle independently of any transport layer:
  - start_run()       validates and launches a background asyncio.Task
  - _execute_run()    drives the graph, writes events to a per-run queue
  - iter_run_events() yields from the queue; safe to call from any transport

The HTTP connection has no ownership over graph execution.  A client
disconnect abandons the queue consumer but the Task continues to completion,
writing the final state to the LangGraph checkpoint.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from langgraph.types import Command

from solidcue.agents.configs.loader import load_agent
from solidcue.agents.configs.schema import AgentConfig
from solidcue.core.router_graph.builder import build_async_router_graph
from solidcue.core.router_graph.router_node._shared import (
    clear_runtime_router_provider_config,
    set_runtime_router_provider_config,
)
from solidcue.core.graph.builder import build_agent_graph, build_async_agent_graph
from solidcue.core.graph_node.final_output_node import (
    prepare_final_output_stream,
    resolve_final_output,
)
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.debug import log_state
from solidcue.core.utils.metrics import build_metric, build_metric_state_delta
from solidcue.services.chat_history_service import (
    add_conversation_worked_seconds,
    append_chat_message,
    get_conversation_metadata,
    update_conversation_run_state,
    upsert_conversation,
)
from solidcue.observability import (
    configure_langsmith_tracing_env,
    flush_langfuse,
    get_langfuse_callbacks,
    propagate_langfuse_session,
    start_langfuse_root_span,
    trace_langgraph_invoke,
)
from solidcue.user.loader import load_user_profile
from solidcue.services.thread_service import create_thread_id


# ---------------------------------------------------------------------------
# In-process run state
# ---------------------------------------------------------------------------

_ACTIVE_RUNS: dict[str, dict[str, Any]] = {}   # run_id  → run info
_THREAD_LATEST_RUN: dict[str, str] = {}         # thread_id → latest run_id
_RUN_QUEUES: dict[str, asyncio.Queue] = {}      # run_id  → event queue
_RUN_TASKS: dict[str, asyncio.Task] = {}        # run_id  → background Task
_BACKGROUND_TASKS: set[asyncio.Task] = set()    # keeps tasks from being GC'd

_QUEUE_SENTINEL: object = object()              # marks a closed queue


# ---------------------------------------------------------------------------
# Run status
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_run_status(
    *,
    thread_id: str,
    status: str,
    run_id: str,
    agent_key: str,
    error: str | None = None,
) -> None:
    entry = {
        "thread_id": thread_id,
        "run_id": run_id,
        "agent_key": agent_key,
        "status": status,
        "error": error,
        "updated_at": _utc_now_iso(),
    }
    _ACTIVE_RUNS[run_id] = entry
    _THREAD_LATEST_RUN[thread_id] = run_id


def get_thread_run_status(thread_id: str) -> dict[str, Any]:
    run_id = _THREAD_LATEST_RUN.get(thread_id)
    if run_id:
        current = _ACTIVE_RUNS.get(run_id)
        if isinstance(current, dict):
            return dict(current)
    return {
        "thread_id": thread_id,
        "run_id": None,
        "agent_key": None,
        "status": "idle",
        "error": None,
        "updated_at": None,
    }


def cancel_run(run_id: str) -> bool:
    """Cancel a running background Task by run_id.

    Returns True if the task was found and cancellation was requested.
    The Task handles CancelledError and sets status to 'cancelled'.
    """
    task = _RUN_TASKS.get(run_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True


async def is_thread_resumable(thread_id: str) -> dict[str, Any]:
    """Return whether a thread has unfinished graph execution that can be continued.

    Reads snapshot.next from the checkpointer — if non-empty the graph stopped
    before END and can be continued via the continue path (no user_input).
    """
    try:
        graph = await build_async_agent_graph()
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return {"resumable": False, "next_nodes": []}
    if not snapshot:
        return {"resumable": False, "next_nodes": []}
    next_nodes = list(getattr(snapshot, "next", None) or [])
    return {"resumable": len(next_nodes) > 0, "next_nodes": next_nodes}


# ---------------------------------------------------------------------------
# Runtime helpers (no HTTP / transport knowledge)
# ---------------------------------------------------------------------------

def _build_run_config(*, agent_key: str, profile_data: dict, debug: bool) -> dict:
    metadata: dict[str, Any] = {"agent_key": agent_key, "debug": debug}
    for key in ("location", "timezone"):
        value = profile_data.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    run_config: dict[str, Any] = {
        "run_name": f"solidcue:{agent_key}",
        "tags": ["solidcue", "langgraph", f"agent:{agent_key}"],
        "metadata": metadata,
    }
    callbacks = get_langfuse_callbacks()
    if callbacks:
        run_config["callbacks"] = callbacks
    return run_config


def _load_agent_runtime(
    *, agent_key: str, debug: bool
) -> tuple[AgentConfig, dict[str, Any], dict[str, Any]]:
    agent = load_agent(agent_key)
    profile_data = load_user_profile().model_dump(exclude_none=True)
    configure_langsmith_tracing_env()
    run_config = _build_run_config(
        agent_key=agent.agent_key, profile_data=profile_data, debug=debug
    )
    return agent, profile_data, run_config


def _build_initial_state(
    *,
    agent_key: str,
    thread_id: str,
    conversation_id: str,
    user_input: str,
    profile_data: dict[str, Any],
) -> AgentState:
    return {
        "agent_key": agent_key,
        "thread_id": thread_id,
        "conversation_id": conversation_id,
        "user_input": user_input,
        "config": profile_data,
        "max_retries": 10,
    }


def _build_router_initial_state(
    *,
    thread_id: str,
    conversation_id: str,
    user_input: str,
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "conversation_id": conversation_id,
        "user_input": user_input,
        "config": {},
        "max_retries": 10,
    }


def _resolve_conversation_id(thread_id: str, conversation_id: str | None = None) -> str:
    if isinstance(conversation_id, str) and conversation_id.strip():
        return conversation_id.strip()
    return thread_id


def _extract_token_usage(node_delta: dict[str, Any]) -> dict[str, int] | None:
    for value in node_delta.values():
        msgs = value if isinstance(value, list) else [value]
        for msg in msgs:
            usage = getattr(msg, "usage_metadata", None)
            if isinstance(usage, dict) and usage:
                return {
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                }
    return None


def _first_output(result: dict[str, Any]) -> str | None:
    for key in ("final_output", "final_response", "synthesis_draft", "draft_output"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _persist_conversation_worked_seconds(
    *,
    conversation_id: str | None,
    thread_id: str,
    agent_key: str,
    started_at: float,
) -> None:
    resolved_conversation_id = (
        conversation_id.strip()
        if isinstance(conversation_id, str) and conversation_id.strip()
        else thread_id
    )
    elapsed_seconds = max(0, math.ceil(time.perf_counter() - started_at))
    add_conversation_worked_seconds(
        conversation_id=resolved_conversation_id,
        worked_seconds=elapsed_seconds,
        agent_key=agent_key,
    )


def _persist_conversation_run_state(
    *,
    conversation_id: str | None,
    thread_id: str,
    run_id: str,
    agent_key: str,
    status: str,
) -> None:
    resolved_conversation_id = (
        conversation_id.strip()
        if isinstance(conversation_id, str) and conversation_id.strip()
        else thread_id
    )
    update_conversation_run_state(
        conversation_id=resolved_conversation_id,
        agent_key=agent_key,
        last_thread_id=thread_id,
        last_run_id=run_id,
        last_run_status=status,
    )


def _invoke_graph(
    *, graph: Any, input_payload: Any, run_config: dict[str, Any], debug: bool
) -> Any:
    if not debug:
        return graph.invoke(input_payload, config=run_config)
    for update in graph.stream(input_payload, config=run_config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_name, node_delta in update.items():
            if isinstance(node_delta, dict):
                log_state(str(node_name), node_delta)
    snapshot = graph.get_state(run_config)
    return snapshot.values


def _append_resume_chat_history(
    *, graph: Any, run_config: dict[str, Any], resume_value: str
) -> None:
    conversation_id = run_config.get("metadata", {}).get("conversation_id")
    if not resume_value.strip():
        return
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        return
    try:
        append_chat_message(
            conversation_id=conversation_id,
            role="user",
            content=resume_value,
        )
    except Exception:
        return


async def _async_append_resume_chat_history(
    *, graph: Any, run_config: dict[str, Any], resume_value: str
) -> None:
    conversation_id = run_config.get("metadata", {}).get("conversation_id")
    if not resume_value.strip():
        return
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        return
    try:
        append_chat_message(
            conversation_id=conversation_id,
            role="user",
            content=resume_value,
        )
    except Exception:
        return


async def _async_guard_against_mid_graph_rerun(
    graph: Any, run_config: dict[str, Any]
) -> None:
    try:
        snapshot = await graph.aget_state(run_config)
    except Exception:
        return
    if not snapshot:
        return
    next_nodes = getattr(snapshot, "next", None)
    if next_nodes:
        raise ValueError(
            f"Thread has unfinished execution at {list(next_nodes)}. "
            "Use the continue path (no user_input) to resume, or start a new thread."
        )


def _resolve_routed_agent_key(state: dict[str, Any]) -> str | None:
    agent_key = state.get("target_agent_key")
    if isinstance(agent_key, str) and agent_key.strip():
        return agent_key.strip()

    handoff = state.get("handoff")
    if isinstance(handoff, dict):
        routed = handoff.get("target_agent_key")
        if isinstance(routed, str) and routed.strip():
            return routed.strip()

    return None


# ---------------------------------------------------------------------------
# Background execution task
# ---------------------------------------------------------------------------

async def _execute_run(
    *,
    run_id: str,
    thread_id: str,
    agent_key: str,
    graph: Any,
    payload: Any,
    run_config: dict[str, Any],
    started_at: float,
) -> None:
    """Drive the graph to completion and write events to the run queue.

    This is a background Task — it has no reference to any HTTP connection
    or async generator consumer.  It writes to ``_RUN_QUEUES[run_id]`` and
    always closes the queue (via sentinel) in the finally block.
    """
    q: asyncio.Queue = _RUN_QUEUES[run_id]

    async def emit(event: dict) -> None:
        await q.put(event)

    try:
        with start_langfuse_root_span(
            name="solidcue.langgraph.execute_run",
            input_payload={"thread_id": thread_id, "agent_key": agent_key},
        ):
            with propagate_langfuse_session(session_id=thread_id):
                await emit(
                    {
                        "event": "start",
                        "data": {
                            "thread_id": thread_id,
                            "run_id": run_id,
                            "agent_key": agent_key,
                        },
                    }
                )

                interrupted = False
                async for update in graph.astream(
                    payload, config=run_config, stream_mode="updates"
                ):
                    if not isinstance(update, dict):
                        continue

                    raw_interrupt = update.get("__interrupt__")
                    if raw_interrupt is not None:
                        value = None
                        if isinstance(raw_interrupt, (list, tuple)) and raw_interrupt:
                            value = getattr(raw_interrupt[0], "value", None)
                        interrupted = True
                        _set_run_status(
                            thread_id=thread_id,
                            status="interrupted",
                            run_id=run_id,
                            agent_key=agent_key,
                        )
                        _persist_conversation_run_state(
                            conversation_id=run_config.get("metadata", {}).get("conversation_id"),
                            thread_id=thread_id,
                            run_id=run_id,
                            agent_key=agent_key,
                            status="interrupted",
                        )
                        _persist_conversation_worked_seconds(
                            conversation_id=run_config.get("metadata", {}).get("conversation_id"),
                            thread_id=thread_id,
                            agent_key=agent_key,
                            started_at=started_at,
                        )
                        await emit(
                            {
                                "event": "interrupt",
                                "data": {
                                    "thread_id": thread_id,
                                    "interrupt": value if isinstance(value, dict) else {},
                                },
                            }
                        )
                        continue

                    for node_name, node_delta in update.items():
                        if str(node_name) == "final_output":
                            continue
                        phase = (
                            node_delta.get("phase")
                            if isinstance(node_delta, dict)
                            else None
                        )
                        tokens = (
                            _extract_token_usage(node_delta)
                            if isinstance(node_delta, dict)
                            else None
                        )
                        await emit(
                            {
                                "event": "node",
                                "data": {
                                    "node": str(node_name),
                                    "phase": phase,
                                    "tokens": tokens,
                                },
                            }
                        )

                if interrupted:
                    return

                snapshot = await graph.aget_state(run_config)
                values = getattr(snapshot, "values", None)
                state: dict[str, Any] = values if isinstance(values, dict) else {}
                phase = state.get("phase")
                await emit(
                    {
                        "event": "node",
                        "data": {
                            "node": "final_output",
                            "phase": phase if isinstance(phase, str) else None,
                            "tokens": None,
                        },
                    }
                )

                output = ""
                metric_final_output: dict[str, Any] = {}
                message_started = False
                prepared_stream = prepare_final_output_stream(state)
                if prepared_stream:
                    provider, messages = prepared_stream
                    await emit(
                        {
                            "event": "message_start",
                            "data": {
                                "thread_id": thread_id,
                                "phase": phase if isinstance(phase, str) else None,
                            },
                        }
                    )
                    message_started = True
                    started = time.perf_counter()
                    chunks: list[str] = []
                    try:
                        async for chunk in provider.async_stream_generate(messages):
                            if not isinstance(chunk, str) or not chunk:
                                continue
                            chunks.append(chunk)
                            await emit(
                                {
                                    "event": "message_delta",
                                    "data": {
                                        "thread_id": thread_id,
                                        "delta": chunk,
                                    },
                                }
                            )
                    except Exception:
                        chunks = [resolve_final_output(state)]
                        if chunks[0]:
                            await emit(
                                {
                                    "event": "message_delta",
                                    "data": {
                                        "thread_id": thread_id,
                                        "delta": chunks[0],
                                    },
                                }
                            )
                    output = "".join(chunks).strip()
                    elapsed_s = time.perf_counter() - started
                    provider_usage = (
                        provider.get_last_usage()
                        if callable(getattr(provider, "get_last_usage", None))
                        else {}
                    )
                    metric_final_output = build_metric(
                        provider_usage if isinstance(provider_usage, dict) else {},
                        elapsed_s,
                        str(getattr(provider, "model", "") or ""),
                    )

                if not output:
                    output = resolve_final_output(state)
                    if not message_started:
                        await emit(
                            {
                                "event": "message_start",
                                "data": {
                                    "thread_id": thread_id,
                                    "phase": phase if isinstance(phase, str) else None,
                                },
                            }
                        )
                    if output:
                        await emit(
                            {
                                "event": "message_delta",
                                "data": {"thread_id": thread_id, "delta": output},
                            }
                        )

                conversation_id = run_config.get("metadata", {}).get("conversation_id")
                append_chat_message(
                    conversation_id=conversation_id if isinstance(conversation_id, str) else thread_id,
                    role="assistant",
                    content=output,
                    agent_key=agent_key,
                )
                await graph.aupdate_state(
                    run_config,
                    {
                        "final_response": output,
                        **build_metric_state_delta(
                            "final_output", "metric_final_output", metric_final_output
                        ),
                    },
                    as_node="final_output",
                )

                _set_run_status(
                    thread_id=thread_id,
                    status="completed",
                    run_id=run_id,
                    agent_key=agent_key,
                )
                _persist_conversation_run_state(
                    conversation_id=conversation_id
                    if isinstance(conversation_id, str)
                    else None,
                    thread_id=thread_id,
                    run_id=run_id,
                    agent_key=agent_key,
                    status="completed",
                )
                _persist_conversation_worked_seconds(
                    conversation_id=conversation_id
                    if isinstance(conversation_id, str)
                    else None,
                    thread_id=thread_id,
                    agent_key=agent_key,
                    started_at=started_at,
                )
                await emit(
                    {
                        "event": "completed",
                        "data": {
                            "thread_id": thread_id,
                            "output": (
                                output
                                or _first_output(state)
                                or "No final response generated."
                            ),
                            "phase": phase if isinstance(phase, str) else None,
                        },
                    }
                )

    except asyncio.CancelledError:
        _set_run_status(
            thread_id=thread_id,
            status="cancelled",
            run_id=run_id,
            agent_key=agent_key,
        )
        _persist_conversation_run_state(
            conversation_id=run_config.get("metadata", {}).get("conversation_id"),
            thread_id=thread_id,
            run_id=run_id,
            agent_key=agent_key,
            status="cancelled",
        )
        _persist_conversation_worked_seconds(
            conversation_id=run_config.get("metadata", {}).get("conversation_id"),
            thread_id=thread_id,
            agent_key=agent_key,
            started_at=started_at,
        )
        await q.put({"event": "cancelled", "data": {"thread_id": thread_id, "run_id": run_id}})
    except Exception as error:
        _set_run_status(
            thread_id=thread_id,
            status="error",
            run_id=run_id,
            agent_key=agent_key,
            error=str(error),
        )
        _persist_conversation_run_state(
            conversation_id=run_config.get("metadata", {}).get("conversation_id"),
            thread_id=thread_id,
            run_id=run_id,
            agent_key=agent_key,
            status="error",
        )
        _persist_conversation_worked_seconds(
            conversation_id=run_config.get("metadata", {}).get("conversation_id"),
            thread_id=thread_id,
            agent_key=agent_key,
            started_at=started_at,
        )
        await q.put({"event": "error", "data": {"message": str(error)}})
    finally:
        _RUN_TASKS.pop(run_id, None)
        await q.put(_QUEUE_SENTINEL)
        flush_langfuse()


# ---------------------------------------------------------------------------
# Public streaming API
# ---------------------------------------------------------------------------

async def start_run(
    *,
    agent_key: str,
    thread_id: str | None = None,
    conversation_id: str | None = None,
    user_input: str | None = None,
    resume_value: str | None = None,
    record_user_message: bool = True,
) -> str:
    """Validate the request, build the payload, and launch a background execution Task.

    Returns the ``run_id`` immediately.  The caller does not wait for the graph
    to finish — it connects to the event stream via ``iter_run_events(run_id)``.
    """
    run_id = str(uuid4())
    thread_id = thread_id or create_thread_id()
    agent, profile_data, run_config = _load_agent_runtime(agent_key=agent_key, debug=False)
    graph = await build_async_agent_graph(streaming_final_output=True)
    run_config["configurable"] = {"thread_id": thread_id}
    resolved_conversation_id = _resolve_conversation_id(thread_id, conversation_id)
    run_config.setdefault("metadata", {})["conversation_id"] = resolved_conversation_id
    upsert_conversation(
        conversation_id=resolved_conversation_id,
        agent_key=agent.agent_key,
        last_thread_id=thread_id,
        last_run_id=run_id,
        last_run_status="running",
    )

    # Pre-flight: resolve payload and validate state BEFORE starting the task
    # so any ValueError surfaces as an HTTP error, not a queue error.
    if resume_value is not None:
        await _async_append_resume_chat_history(
            graph=graph, run_config=run_config, resume_value=resume_value
        )
        payload: Any = Command(resume=resume_value)
    elif user_input is not None:
        await _async_guard_against_mid_graph_rerun(graph, run_config)
        if record_user_message:
            append_chat_message(
                conversation_id=resolved_conversation_id,
                role="user",
                content=user_input,
                agent_key=agent.agent_key,
            )
        payload = _build_initial_state(
            agent_key=agent.agent_key,
            thread_id=thread_id,
            conversation_id=resolved_conversation_id,
            user_input=user_input,
            profile_data=profile_data,
        )
    else:
        payload = None  # continue from checkpoint

    q: asyncio.Queue = asyncio.Queue()
    _RUN_QUEUES[run_id] = q
    _set_run_status(
        thread_id=thread_id,
        status="running",
        run_id=run_id,
        agent_key=agent.agent_key,
    )

    task = asyncio.create_task(
        _execute_run(
            run_id=run_id,
            thread_id=thread_id,
            agent_key=agent.agent_key,
            graph=graph,
            payload=payload,
            run_config=run_config,
            started_at=time.perf_counter(),
        )
    )
    _RUN_TASKS[run_id] = task
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    return run_id


async def iter_run_events(run_id: str) -> AsyncIterator[dict[str, Any]]:
    """Yield events from the run queue until the run closes.

    Transport-agnostic: SSE, WebSocket, Telegram, or any other consumer can
    call this.  If the consumer disconnects, the background Task keeps running
    and the queue buffers remaining events.
    """
    q = _RUN_QUEUES.get(run_id)
    if q is None:
        return
    while True:
        event = await q.get()
        if event is _QUEUE_SENTINEL:
            break
        yield event


# ---------------------------------------------------------------------------
# Sync (non-streaming) execution paths — used by /run and /resume endpoints
# ---------------------------------------------------------------------------

def run_agent_step(
    *,
    agent_key: str,
    thread_id: str,
    conversation_id: str | None = None,
    debug: bool = False,
    user_input: str | None = None,
    resume_value: str | None = None,
    record_user_message: bool = True,
) -> tuple[AgentConfig, Any]:
    agent, profile_data, run_config = _load_agent_runtime(agent_key=agent_key, debug=debug)
    graph = build_agent_graph()
    run_config["configurable"] = {"thread_id": thread_id}
    resolved_conversation_id = _resolve_conversation_id(thread_id, conversation_id)
    run_config.setdefault("metadata", {})["conversation_id"] = resolved_conversation_id
    upsert_conversation(
        conversation_id=resolved_conversation_id,
        agent_key=agent.agent_key,
    )
    started_at = time.perf_counter()

    if resume_value is not None:
        _append_resume_chat_history(graph=graph, run_config=run_config, resume_value=resume_value)
        try:
            with start_langfuse_root_span(
                name="solidcue.langgraph.run_agent_step.resume",
                input_payload={"thread_id": thread_id, "agent_key": agent.agent_key},
            ):
                with propagate_langfuse_session(session_id=thread_id):
                    result = trace_langgraph_invoke(
                        span_name="solidcue.langgraph.run_agent_step.resume",
                        attributes={
                            "solidcue.agent_key": agent.agent_key,
                            "solidcue.thread_id": thread_id,
                            "solidcue.debug": debug,
                        },
                        invoke=lambda: _invoke_graph(
                            graph=graph,
                            input_payload=Command(resume=resume_value),
                            run_config=run_config,
                            debug=debug,
                        ),
                    )
                    output = _first_output(result) if isinstance(result, dict) else None
                    if isinstance(output, str) and output.strip():
                        append_chat_message(
                            conversation_id=resolved_conversation_id,
                            role="assistant",
                            content=output,
                            agent_key=agent.agent_key,
                        )
                    _persist_conversation_worked_seconds(
                        conversation_id=resolved_conversation_id,
                        thread_id=thread_id,
                        agent_key=agent.agent_key,
                        started_at=started_at,
                    )
                    return (
                        agent,
                        result,
                    )
        finally:
            flush_langfuse()

    if user_input is None:
        raise ValueError("user_input is required for initial run")

    if record_user_message:
        append_chat_message(
            conversation_id=resolved_conversation_id,
            role="user",
            content=user_input,
            agent_key=agent.agent_key,
        )
    state = _build_initial_state(
        agent_key=agent.agent_key,
        thread_id=thread_id,
        conversation_id=resolved_conversation_id,
        user_input=user_input,
        profile_data=profile_data,
    )
    try:
        with start_langfuse_root_span(
            name="solidcue.langgraph.run_agent_step.initial",
            input_payload={"thread_id": thread_id, "agent_key": agent.agent_key},
        ):
            with propagate_langfuse_session(session_id=thread_id):
                result = trace_langgraph_invoke(
                    span_name="solidcue.langgraph.run_agent_step.initial",
                    attributes={
                        "solidcue.agent_key": agent.agent_key,
                        "solidcue.thread_id": thread_id,
                        "solidcue.debug": debug,
                    },
                    invoke=lambda: _invoke_graph(
                        graph=graph,
                        input_payload=state,
                        run_config=run_config,
                        debug=debug,
                    ),
                )
                output = _first_output(result) if isinstance(result, dict) else None
                if isinstance(output, str) and output.strip():
                    append_chat_message(
                        conversation_id=resolved_conversation_id,
                        role="assistant",
                        content=output,
                        agent_key=agent.agent_key,
                    )
                _persist_conversation_worked_seconds(
                    conversation_id=resolved_conversation_id,
                    thread_id=thread_id,
                    agent_key=agent.agent_key,
                    started_at=started_at,
                )
                return (
                    agent,
                    result,
                )
    finally:
        flush_langfuse()


def run_agent(
    agent_key: str,
    user_input: str,
    thread_id: str,
    conversation_id: str | None = None,
    debug: bool = False,
    record_user_message: bool = True,
) -> tuple[AgentConfig, AgentState]:
    """Blocking full-graph run.  Returns when the graph reaches END."""
    agent, profile_data, run_config = _load_agent_runtime(agent_key=agent_key, debug=debug)
    resolved_conversation_id = _resolve_conversation_id(thread_id, conversation_id)
    run_config.setdefault("metadata", {})["conversation_id"] = resolved_conversation_id
    upsert_conversation(
        conversation_id=resolved_conversation_id,
        agent_key=agent.agent_key,
    )
    if record_user_message:
        append_chat_message(
            conversation_id=resolved_conversation_id,
            role="user",
            content=user_input,
            agent_key=agent.agent_key,
        )
    state = _build_initial_state(
        agent_key=agent.agent_key,
        thread_id=thread_id,
        conversation_id=resolved_conversation_id,
        user_input=user_input,
        profile_data=profile_data,
    )
    graph = build_agent_graph()
    run_config["configurable"] = {"thread_id": thread_id}
    started_at = time.perf_counter()
    try:
        with start_langfuse_root_span(
            name="solidcue.langgraph.run_agent",
            input_payload={"thread_id": thread_id, "agent_key": agent.agent_key},
        ):
            with propagate_langfuse_session(session_id=thread_id):
                result = cast(
                    AgentState,
                    trace_langgraph_invoke(
                        span_name="solidcue.langgraph.run_agent",
                        attributes={
                            "solidcue.agent_key": agent.agent_key,
                            "solidcue.thread_id": thread_id,
                            "solidcue.debug": debug,
                        },
                        invoke=lambda: _invoke_graph(
                            graph=graph,
                            input_payload=state,
                            run_config=run_config,
                            debug=debug,
                        ),
                    ),
                )
                output = _first_output(result)
                if isinstance(output, str) and output.strip():
                    append_chat_message(
                        conversation_id=resolved_conversation_id,
                        role="assistant",
                        content=output,
                        agent_key=agent.agent_key,
                    )
                _persist_conversation_worked_seconds(
                    conversation_id=resolved_conversation_id,
                    thread_id=thread_id,
                    agent_key=agent.agent_key,
                    started_at=started_at,
                )
    finally:
        flush_langfuse()
    return agent, result


async def stream_agent_events(
    *,
    agent_key: str,
    thread_id: str,
    conversation_id: str | None = None,
    user_input: str | None = None,
    resume_value: str | None = None,
    record_user_message: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Yield transport-agnostic stream events for a run or resume.

    Wraps ``start_run`` + ``iter_run_events``.  Callers that need the old
    single-generator interface (e.g. tests) can use this without knowing about
    the background-task/queue internals.
    """
    run_id = await start_run(
        agent_key=agent_key,
        thread_id=thread_id,
        conversation_id=conversation_id,
        user_input=user_input,
        resume_value=resume_value,
        record_user_message=record_user_message,
    )
    async for event in iter_run_events(run_id):
        yield event


async def stream_router_chat_events(
    *,
    thread_id: str | None = None,
    conversation_id: str | None = None,
    user_input: str | None = None,
    resume_value: str | None = None,
    router_provider_config: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the user-facing router graph, then hand off to a selected agent graph.

    The router graph owns intent detection. When it returns a task handoff, the
    runtime starts a second graph using the same conversation_id but a fresh
    thread_id for the target agent.
    """
    if resume_value is not None:
        resolved_conversation_id = _resolve_conversation_id(thread_id or create_thread_id(), conversation_id)
        metadata = get_conversation_metadata(resolved_conversation_id)
        agent_key = (
            metadata.get("agent_key")
            if isinstance(metadata, dict)
            and isinstance(metadata.get("agent_key"), str)
            and metadata.get("agent_key", "").strip()
            else None
        )
        latest_thread_id = (
            metadata.get("last_thread_id")
            if isinstance(metadata, dict)
            and isinstance(metadata.get("last_thread_id"), str)
            and metadata.get("last_thread_id", "").strip()
            else None
        )
        if not agent_key or not latest_thread_id:
            raise ValueError("No routed agent thread found for this conversation")
        async for event in stream_agent_events(
            agent_key=agent_key,
            thread_id=latest_thread_id,
            conversation_id=resolved_conversation_id,
            resume_value=resume_value,
            record_user_message=False,
        ):
            yield event
        return

    if user_input is None:
        raise ValueError("user_input is required")

    router_thread_id = thread_id or create_thread_id()
    resolved_conversation_id = _resolve_conversation_id(router_thread_id, conversation_id)
    append_chat_message(
        conversation_id=resolved_conversation_id,
        role="user",
        content=user_input,
        agent_key="router",
    )

    configure_langsmith_tracing_env()
    graph = await build_async_router_graph()
    run_config: dict[str, Any] = {
        "configurable": {"thread_id": router_thread_id},
        "metadata": {
            "conversation_id": resolved_conversation_id,
            "agent_key": "router",
        },
    }

    with start_langfuse_root_span(
        name="solidcue.langgraph.router_run",
        input_payload={"thread_id": router_thread_id, "conversation_id": resolved_conversation_id},
    ):
        with propagate_langfuse_session(session_id=router_thread_id):
            set_runtime_router_provider_config(router_thread_id, router_provider_config)
            yield {
                "event": "start",
                "data": {
                    "thread_id": router_thread_id,
                    "conversation_id": resolved_conversation_id,
                    "agent_key": "router",
                },
            }
            try:
                async for update in graph.astream(
                    _build_router_initial_state(
                        thread_id=router_thread_id,
                        conversation_id=resolved_conversation_id,
                        user_input=user_input,
                    ),
                    config=run_config,
                    stream_mode="updates",
                ):
                    if not isinstance(update, dict):
                        continue
                    for node_name, node_delta in update.items():
                        if isinstance(node_delta, dict):
                            await asyncio.sleep(0)
                            yield {
                                "event": "node",
                                "data": {
                                    "node": str(node_name),
                                    "phase": node_delta.get("router_intent"),
                                    "tokens": None,
                                },
                            }

                snapshot = await graph.aget_state(run_config)
                values = getattr(snapshot, "values", None)
                state: dict[str, Any] = values if isinstance(values, dict) else {}
                output = _first_output(state) or ""
                handoff = state.get("handoff") if isinstance(state.get("handoff"), dict) else {}
                target_agent_key = _resolve_routed_agent_key(state)
                handoff_action = handoff.get("action") if isinstance(handoff, dict) else None

                if isinstance(target_agent_key, str) and target_agent_key.strip() and handoff_action == "route_agent":
                    agent_thread_id = create_thread_id()
                    upsert_conversation(
                        conversation_id=resolved_conversation_id,
                        agent_key=target_agent_key,
                        last_thread_id=agent_thread_id,
                        last_run_status="running",
                    )
                    yield {
                        "event": "handoff",
                        "data": {
                            "thread_id": router_thread_id,
                            "conversation_id": resolved_conversation_id,
                            "target_agent_key": target_agent_key,
                            "agent_thread_id": agent_thread_id,
                        },
                    }
                    async for event in stream_agent_events(
                        agent_key=target_agent_key,
                        thread_id=agent_thread_id,
                        conversation_id=resolved_conversation_id,
                        user_input=str(handoff.get("task_input") or user_input),
                        record_user_message=False,
                    ):
                        yield event
                    return

                if not output and handoff_action == "create_agent":
                    output = "I can help create a new agent. Tell me the agent name, description, and tools."

                if output:
                    append_chat_message(
                        conversation_id=resolved_conversation_id,
                        role="assistant",
                        content=output,
                        agent_key="router",
                    )
                    yield {
                        "event": "message_start",
                        "data": {
                            "thread_id": router_thread_id,
                            "conversation_id": resolved_conversation_id,
                            "phase": state.get("router_intent"),
                        },
                    }
                    yield {
                        "event": "message_delta",
                        "data": {
                            "thread_id": router_thread_id,
                            "delta": output,
                        },
                    }

                yield {
                    "event": "completed",
                    "data": {
                        "thread_id": router_thread_id,
                        "conversation_id": resolved_conversation_id,
                        "output": output,
                        "phase": state.get("router_intent"),
                    },
                }
            finally:
                clear_runtime_router_provider_config(router_thread_id)
