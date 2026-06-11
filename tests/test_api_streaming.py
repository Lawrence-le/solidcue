from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from solidcue.services.run_engine import (
    get_thread_run_status,
    stream_agent_events,
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
    async def astream(self, _payload, config=None, stream_mode="updates"):
        yield {"intent_router": {"router_intent": "task"}}

    async def aget_state(self, _config):
        return SimpleNamespace(
            values={
                "router_intent": "task",
                "router_next": "handoff",
                "target_agent_key": "resume_builder",
                "handoff": {
                    "action": "route_agent",
                    "task_input": "build my resume",
                },
            }
        )


@pytest.mark.asyncio
async def test_stream_agent_events_streams_final_output_and_persists_state(
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
        async for event in stream_agent_events(
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
async def test_stream_agent_events_propagates_langfuse_session_id(
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
        async for event in stream_agent_events(
            agent_key="agent-1", thread_id="thread-x2", user_input="hello"
        )
    ]

    assert events[-1]["event"] == "completed"
    assert captured.get("session_id") == "thread-x2"
    assert captured.get("entered") is True
    assert captured.get("exited") is True
    assert captured.get("flushed") is True


@pytest.mark.asyncio
async def test_stream_router_chat_events_hands_off_to_agent(
    monkeypatch,
) -> None:
    router_graph = _RouterGraph()
    captured: dict[str, object] = {}

    async def _fake_stream_agent_events(**kwargs):
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
        return router_graph

    monkeypatch.setattr(
        "solidcue.services.run_engine.configure_langsmith_tracing_env", lambda: None
    )
    monkeypatch.setattr(
        "solidcue.services.run_engine.build_async_router_graph",
        _fake_build_async_router_graph,
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
        "solidcue.services.run_engine.stream_agent_events",
        _fake_stream_agent_events,
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

    assert events[0]["event"] == "start"
    assert any(event["event"] == "handoff" for event in events)
    assert events[-1]["event"] == "completed"
    assert captured["agent_kwargs"]["agent_key"] == "resume_builder"
    assert captured["agent_kwargs"]["conversation_id"] == "conversation-1"
    assert captured["agent_kwargs"]["record_user_message"] is False
