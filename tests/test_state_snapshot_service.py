import sqlite3

from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    build_state_snapshot,
    delete_thread_state,
    get_latest_thread_id,
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


import pytest

@pytest.mark.asyncio
async def test_build_live_state_snapshot_with_selected_keys(monkeypatch) -> None:
    class _Snapshot:
        values = {"decision": {"action": "respond"}, "retry_reason": "x"}
        tasks = []
        next = []

    class _Graph:
        async def aget_state(self, _config):
            return _Snapshot()

    async def _fake_build_async_agent_graph(**_kwargs):
        return _Graph()

    monkeypatch.setattr(
        "solidcue.services.state_snapshot_service.build_async_agent_graph",
        _fake_build_async_agent_graph,
    )
    snapshot = await build_live_state_snapshot(thread_id="t1", keys=["decision"], include_all=False)
    assert snapshot == {"decision": {"action": "respond"}}


def test_get_latest_thread_id_from_checkpoint_db(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, metadata BLOB)"
    )
    cur.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("thread-old", "", "cp-1", None, "msgpack", b"", b""),
    )
    cur.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("thread-new", "", "cp-2", None, "msgpack", b"", b""),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))
    assert get_latest_thread_id() == "thread-new"


def test_delete_thread_state_removes_rows_from_all_thread_tables(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint BLOB)")
    cur.execute("CREATE TABLE writes (thread_id TEXT, payload BLOB)")
    cur.execute("CREATE TABLE misc (id TEXT, payload BLOB)")
    cur.execute("INSERT INTO checkpoints (thread_id, checkpoint) VALUES (?, ?)", ("thread-1", b"a"))
    cur.execute("INSERT INTO checkpoints (thread_id, checkpoint) VALUES (?, ?)", ("thread-2", b"b"))
    cur.execute("INSERT INTO writes (thread_id, payload) VALUES (?, ?)", ("thread-1", b"c"))
    cur.execute("INSERT INTO misc (id, payload) VALUES (?, ?)", ("thread-1", b"d"))
    conn.commit()
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    assert delete_thread_state("thread-1") is True

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("thread-1",))
    assert cur.fetchone() == (0,)
    cur.execute("SELECT COUNT(*) FROM writes WHERE thread_id = ?", ("thread-1",))
    assert cur.fetchone() == (0,)
    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("thread-2",))
    assert cur.fetchone() == (1,)
    cur.execute("SELECT COUNT(*) FROM misc WHERE id = ?", ("thread-1",))
    assert cur.fetchone() == (1,)
    conn.close()


def test_delete_thread_state_returns_false_when_thread_missing(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint BLOB)")
    conn.commit()
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))
    assert delete_thread_state("thread-missing") is False
