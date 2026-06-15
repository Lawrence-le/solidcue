"""list_threads should hide orchestrated sub-agent (worker) runs from the sidebar."""

from __future__ import annotations

import pytest

import solidcue.api.routes.state as state_routes
from solidcue.api.routes.state import list_threads


def _make_thread(thread_id: str, conversation_id: str, agent_key: str | None = None) -> dict:
    return {
        "thread_id": thread_id,
        "metadata": {"conversation_id": conversation_id, "agent_key": agent_key},
    }


@pytest.mark.asyncio
async def test_list_threads_excludes_worker_conversations(monkeypatch):
    threads = [
        _make_thread("router-thread", "conversation-1", "router"),
        _make_thread("worker-thread-0", "conversation-1::worker::0", "researcher"),
        _make_thread("worker-thread-1", "conversation-1::worker::1", "writer"),
    ]

    class _FakeThreadsClient:
        async def search(self, *, limit=100):
            return threads

    class _FakeClient:
        threads = _FakeThreadsClient()

    monkeypatch.setattr(state_routes, "get_lg_client", lambda: _FakeClient())

    summaries = await list_threads()

    conversation_ids = [s.conversation_id for s in summaries]
    assert conversation_ids == ["conversation-1"]
    assert all("::worker::" not in cid for cid in conversation_ids)
