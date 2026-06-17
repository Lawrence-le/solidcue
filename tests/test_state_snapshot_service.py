import pytest

from solidcue.services import state_snapshot_service as svc
from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    build_state_snapshot,
    delete_thread_state,
    get_latest_thread_id,
    get_thread_interrupt_payload,
    list_agent_state_keys,
)


def test_list_agent_state_keys_includes_decision_related_keys() -> None:
    keys = list_agent_state_keys()
    assert "decision" in keys
    assert "execution_result" in keys
    assert "tool_call_history" in keys


def test_build_state_snapshot_with_selected_keys_only() -> None:
    snapshot = build_state_snapshot(keys=["decision", "metadata", "unknown"], include_all=False)
    assert set(snapshot.keys()) == {"decision", "metadata"}


def test_build_state_snapshot_with_all_keys() -> None:
    snapshot = build_state_snapshot(include_all=True)
    assert "agent_key" in snapshot
    assert "decision" in snapshot
    assert "retry_reason" in snapshot


@pytest.mark.asyncio
async def test_build_live_state_snapshot_reads_from_sdk(monkeypatch) -> None:
    async def _fake_get_lg_thread_state(_thread_id):
        return {"decision": {"action": "respond"}, "retry_reason": "x"}

    monkeypatch.setattr(svc, "get_lg_thread_state", _fake_get_lg_thread_state)
    snapshot = await build_live_state_snapshot(thread_id="t1", keys=["decision"], include_all=False)
    assert snapshot == {"decision": {"action": "respond"}}


@pytest.mark.asyncio
async def test_get_latest_thread_id_delegates_to_sdk(monkeypatch) -> None:
    async def _fake_latest():
        return "thread-new"

    monkeypatch.setattr(svc, "get_lg_latest_thread_id", _fake_latest)
    assert await get_latest_thread_id() == "thread-new"


@pytest.mark.asyncio
async def test_delete_thread_state_delegates_to_sdk(monkeypatch) -> None:
    calls: list[str] = []

    async def _fake_delete(thread_id):
        calls.append(thread_id)
        return True

    monkeypatch.setattr(svc, "delete_lg_thread", _fake_delete)
    assert await delete_thread_state("thread-1") is True
    assert calls == ["thread-1"]


@pytest.mark.asyncio
async def test_delete_thread_state_rejects_blank() -> None:
    assert await delete_thread_state("   ") is False


@pytest.mark.asyncio
async def test_get_thread_interrupt_payload_delegates_to_sdk(monkeypatch) -> None:
    async def _fake_interrupt(_thread_id):
        return {"kind": "approval", "preview": "do X?"}

    monkeypatch.setattr(svc, "get_lg_thread_interrupt", _fake_interrupt)
    payload = await get_thread_interrupt_payload("t1")
    assert payload == {"kind": "approval", "preview": "do X?"}
