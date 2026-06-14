import asyncio
import contextlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import solidcue.services.run_engine as run_engine
from solidcue.services.run_engine import (
    get_thread_run_status,
    start_run,
    stream_agent_graph_events,
    stream_router_chat_events,
)


class _Provider:
    model = "stream-model"

    def __init__(self) -> None:
        self.last_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cached_tokens": 0,
            "method": "provider_reported",
        }

    async def async_stream_generate(self, _messages, **_kwargs):
        yield "Hello"
        yield " world"

    def get_last_usage(self):
        return self.last_usage


class _Snapshot:
    values = {
        "agent_key": "agent-1",
        "phase": "final",
        "synthesis_draft": "Fallback draft",
    }
    tasks = []
    next = []


class _Graph:
    def __init__(self) -> None:
        self.updated = None

    async def astream(self, _payload, config=None, stream_mode="updates"):
        yield {"planning": {"phase": "source"}}

    async def aget_state(self, _config):
        return _Snapshot()

    async def aupdate_state(self, _config, values, as_node=None, task_id=None):
        self.updated = (values, as_node, task_id)
        return _config


class _RouterGraph:
    async def astream(self, _payload, config=None, stream_mode="updates", version="v1"):
        yield {"type": "updates", "data": {"initialize": {"router_intent": "chat"}}}
        yield {"type": "updates", "data": {"intent_router": {"router_intent": "task"}}}
        yield {"type": "custom", "data": "Sure, "}
        yield {"type": "custom", "data": "I will help you generate a resume based on your request."}

    async def aget_state(self, _config):
        return SimpleNamespace(
            values={
                "router_intent": "task",
                "router_next": "handoff",
                "target_agent_key": "resume_builder",
                "assistant_draft": "Sure, I will help you generate a resume based on your request.",
                "final_response": "Sure, I will help you generate a resume based on your request.",
                "handoff": {
                    "action": "route_agent",
                    "task_input": "build my resume",
                },
            }
        )


@pytest.mark.asyncio
async def test_stream_agent_graph_events_streams_final_output_and_persists_state(
    monkeypatch,
) -> None:
    fake_graph = _Graph()

    monkeypatch.setattr(
        "solidcue.services.run_engine.load_agent",
        lambda _agent_key: SimpleNamespace(agent_key="agent-1"),
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.load_user_profile",
        lambda: SimpleNamespace(model_dump=lambda exclude_none=True: {}),
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.configure_langsmith_tracing_env", lambda: None
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.build_async_agent_graph",
        lambda streaming_final_output=False: fake_graph,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.prepare_final_output_stream",
        lambda state: (_Provider(), [{"role": "user", "content": "hi"}]),
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.append_chat_message",
        lambda **_kwargs: None,
    )

    async def _fake_build_async_agent_graph(*, streaming_final_output=False):
        return fake_graph

    monkeypatch.setattr(
        "solidcue.services.run_engine.build_async_agent_graph",
        _fake_build_async_agent_graph,
    )

    events = [
        event
        async for event in stream_agent_graph_events(
            agent_key="agent-1", thread_id="thread-x1", user_input="hello"
        )
    ]

    event_names = [e["event"] for e in events]
    assert event_names == [
        "start",
        "node",
        "node",
        "message_start",
        "message_delta",
        "message_delta",
        "completed",
    ]
    assert events[-1]["data"]["output"] == "Hello world"
    assert fake_graph.updated is not None
    updated_values, as_node, _task_id = fake_graph.updated
    assert updated_values["final_response"] == "Hello world"
    assert as_node == "final_output"
    status = get_thread_run_status("thread-x1")
    assert status["status"] == "completed"
    assert status["agent_key"] == "agent-1"


@pytest.mark.asyncio
async def test_stream_agent_graph_events_propagates_langfuse_session_id(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    fake_graph = _Graph()

    @contextmanager
    def _propagate_langfuse_session(*, session_id: str | None):
        captured["session_id"] = session_id
        captured["entered"] = True
        try:
            yield
        finally:
            captured["exited"] = True

    @contextmanager
    def _start_langfuse_root_span(**_kwargs):
        yield None

    async def _fake_build_async_agent_graph(*, streaming_final_output=False):
        return fake_graph

    monkeypatch.setattr(
        "solidcue.services.run_engine.load_agent",
        lambda _agent_key: SimpleNamespace(agent_key="agent-1"),
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.load_user_profile",
        lambda: SimpleNamespace(model_dump=lambda exclude_none=True: {}),
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.configure_langsmith_tracing_env", lambda: None
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.build_async_agent_graph",
        _fake_build_async_agent_graph,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.prepare_final_output_stream",
        lambda state: None,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.append_chat_message",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.propagate_langfuse_session",
        _propagate_langfuse_session,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.start_langfuse_root_span",
        _start_langfuse_root_span,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.flush_langfuse",
        lambda: captured.setdefault("flushed", True),
    )

    events = [
        event
        async for event in stream_agent_graph_events(
            agent_key="agent-1", thread_id="thread-x2", user_input="hello"
        )
    ]

    assert events[-1]["event"] == "completed"
    assert captured.get("session_id") == "thread-x2"
    assert captured.get("entered") is True
    assert captured.get("exited") is True
    assert captured.get("flushed") is True


@pytest.mark.asyncio
async def test_start_run_reuses_active_run_for_thread(monkeypatch) -> None:
    fake_graph = _Graph()
    existing_run_id = "run-active"
    pending_task = asyncio.create_task(asyncio.sleep(60))
    pending_queue = asyncio.Queue()

    async def _fake_build_async_agent_graph(*, streaming_final_output=False):
        return fake_graph

    monkeypatch.setattr(
        "solidcue.services.run_engine.load_agent",
        lambda _agent_key: SimpleNamespace(agent_key="agent-1"),
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.load_user_profile",
        lambda: SimpleNamespace(model_dump=lambda exclude_none=True: {}),
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.configure_langsmith_tracing_env", lambda: None
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.build_async_agent_graph",
        _fake_build_async_agent_graph,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.append_chat_message",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.upsert_conversation",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run_engine,
        "_THREAD_LATEST_RUN",
        {"thread-reuse": existing_run_id},
    )
    monkeypatch.setattr(
        run_engine,
        "_ACTIVE_RUNS",
        {
            existing_run_id: {
                "thread_id": "thread-reuse",
                "run_id": existing_run_id,
                "agent_key": "agent-1",
                "status": "running",
                "error": None,
                "updated_at": "now",
            }
        },
    )
    monkeypatch.setattr(
        run_engine,
        "_RUN_TASKS",
        {existing_run_id: pending_task},
    )
    monkeypatch.setattr(
        run_engine,
        "_RUN_QUEUES",
        {existing_run_id: pending_queue},
    )
    monkeypatch.setattr(
        run_engine,
        "_BACKGROUND_TASKS",
        set(),
    )

    try:
        run_id = await start_run(
            agent_key="agent-1",
            thread_id="thread-reuse",
            conversation_id="conversation-reuse",
        )
        assert run_id == existing_run_id
        assert set(run_engine._RUN_TASKS.keys()) == {existing_run_id}
    finally:
        pending_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending_task


@pytest.mark.asyncio
async def test_stream_router_chat_events_reconnects_to_active_run(monkeypatch) -> None:
    existing_run_id = "router-run-active"
    pending_task = asyncio.create_task(asyncio.sleep(60))
    pending_queue = asyncio.Queue()
    await pending_queue.put({"event": "node", "data": {"node": "intent_router", "phase": "task", "tokens": None}})
    await pending_queue.put({"event": "completed", "data": {"thread_id": "thread-router", "output": "done", "phase": "task"}})
    await pending_queue.put(run_engine._QUEUE_SENTINEL)

    monkeypatch.setattr(
        run_engine,
        "_THREAD_LATEST_RUN",
        {"thread-router": existing_run_id},
    )
    monkeypatch.setattr(
        run_engine,
        "_ACTIVE_RUNS",
        {
            existing_run_id: {
                "thread_id": "thread-router",
                "run_id": existing_run_id,
                "agent_key": "router",
                "status": "running",
                "error": None,
                "updated_at": "now",
            }
        },
    )
    monkeypatch.setattr(
        run_engine,
        "_RUN_TASKS",
        {existing_run_id: pending_task},
    )
    monkeypatch.setattr(
        run_engine,
        "_RUN_QUEUES",
        {existing_run_id: pending_queue},
    )
    monkeypatch.setattr(
        run_engine,
        "get_conversation_metadata",
        lambda _conversation_id: {"last_thread_id": "thread-router"},
    )

    try:
        events = [
            event
            async for event in stream_router_chat_events(
                conversation_id="conversation-router"
            )
        ]
        assert [event["event"] for event in events] == ["node", "completed"]
    finally:
        pending_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending_task


@pytest.mark.asyncio
async def test_stream_router_chat_events_continues_checkpoint_without_active_run(
    monkeypatch,
) -> None:
    class _ContinueRouterGraph:
        async def astream(self, _payload, config=None, stream_mode="updates", version="v1"):
            yield {"type": "updates", "data": {"initialize": {"router_intent": "chat"}}}
            yield {"type": "updates", "data": {"intent_router": {"router_intent": "chat"}}}
            yield {"type": "custom", "data": "Continuing from checkpoint."}

        async def aget_state(self, _config):
            return SimpleNamespace(
                values={
                    "router_intent": "chat",
                    "router_next": "final_output",
                    "target_agent_key": "",
                    "assistant_draft": "Continuing from checkpoint.",
                    "final_response": "Continuing from checkpoint.",
                    "handoff": {},
                }
            )

    graph_router = _ContinueRouterGraph()

    async def _fake_build_async_router_graph():
        return graph_router

    monkeypatch.setattr(
        run_engine,
        "get_conversation_metadata",
        lambda _conversation_id: {
            "agent_key": "router",
            "last_thread_id": "thread-router",
        },
    )
    monkeypatch.setattr(
        run_engine,
        "load_user_profile",
        lambda: SimpleNamespace(router_provider=_Provider()),
    )
    monkeypatch.setattr(
        run_engine,
        "build_async_router_graph",
        _fake_build_async_router_graph,
    )
    monkeypatch.setattr(
        run_engine,
        "append_chat_message",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run_engine,
        "upsert_conversation",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run_engine,
        "_THREAD_LATEST_RUN",
        {},
    )
    monkeypatch.setattr(
        run_engine,
        "_ACTIVE_RUNS",
        {},
    )
    monkeypatch.setattr(
        run_engine,
        "_RUN_TASKS",
        {},
    )
    monkeypatch.setattr(
        run_engine,
        "_RUN_QUEUES",
        {},
    )
    monkeypatch.setattr(
        run_engine,
        "_BACKGROUND_TASKS",
        set(),
    )
    monkeypatch.setattr(
        run_engine,
        "stream_agent_graph_events",
        lambda **_kwargs: (_ for _ in ()),
    )

    events = [
        event
        async for event in stream_router_chat_events(conversation_id="conversation-router")
    ]

    event_names = [event["event"] for event in events]
    assert event_names[0] == "start"
    assert "message_start" in event_names
    assert "message_delta" in event_names
    assert event_names[-1] == "completed"


@pytest.mark.asyncio
async def test_stream_router_chat_events_hands_off_to_agent(
    monkeypatch,
) -> None:
    graph_router = _RouterGraph()
    captured: dict[str, object] = {}
    append_calls: list[dict[str, object]] = []

    async def _fake_stream_agent_graph_events(**kwargs):
        captured["agent_kwargs"] = kwargs
        yield {"event": "start", "data": {"thread_id": kwargs["thread_id"]}}
        yield {"event": "completed", "data": {"output": "done"}}

    @contextmanager
    def _start_langfuse_root_span(**_kwargs):
        yield None

    @contextmanager
    def _propagate_langfuse_session(*, session_id: str | None):
        captured["session_id"] = session_id
        yield

    async def _fake_build_async_router_graph():
        return graph_router

    monkeypatch.setattr(
        "solidcue.services.run_engine.configure_langsmith_tracing_env", lambda: None
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.build_async_router_graph",
        _fake_build_async_router_graph,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.append_chat_message",
        lambda **kwargs: append_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.upsert_conversation",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.stream_agent_graph_events",
        _fake_stream_agent_graph_events,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.start_langfuse_root_span",
        _start_langfuse_root_span,
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.propagate_langfuse_session",
        _propagate_langfuse_session,
    )

    events = [
        event
        async for event in stream_router_chat_events(
            thread_id="router-thread",
            conversation_id="conversation-1",
            user_input="build my resume",
        )
    ]

    event_names = [event["event"] for event in events]
    node_names = [event["data"]["node"] for event in events if event["event"] == "node"]
    assert events[0]["event"] == "start"
    assert node_names == ["initialize", "intent_router", "handoff", "final_output"]
    assert event_names.index("message_start") > event_names.index("node")
    assert event_names.index("message_delta") > event_names.index("message_start")
    assert any(event["event"] == "handoff" for event in events)
    assert events[-1]["event"] == "completed"
    assert captured["agent_kwargs"]["agent_key"] == "resume_builder"
    assert captured["agent_kwargs"]["conversation_id"] == "conversation-1"
    assert captured["agent_kwargs"]["record_user_message"] is False
    assert sum(1 for call in append_calls if call.get("role") == "user") == 1
