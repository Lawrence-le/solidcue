import sqlite3

from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    build_state_snapshot,
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


def test_build_live_state_snapshot_with_selected_keys(monkeypatch) -> None:
    class _Snapshot:
        values = {"decision": {"action": "respond"}, "retry_reason": "x"}

    class _Graph:
        def get_state(self, _config):
            return _Snapshot()

    monkeypatch.setattr("solidcue.services.state_snapshot_service.build_agent_graph", lambda: _Graph())
    snapshot = build_live_state_snapshot(thread_id="t1", keys=["decision"], include_all=False)
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
