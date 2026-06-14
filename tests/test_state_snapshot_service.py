import sqlite3

import pytest

from solidcue.services.chat_history_service import (
    append_chat_message,
    get_conversation_metadata,
)
from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    build_state_snapshot,
    delete_conversation_state,
    delete_thread_state,
    get_latest_thread_id,
    is_conversation_resumable,
    list_agent_state_keys,
    load_conversation_metadata,
    load_conversation_snapshot,
    prune_thread_checkpoints,
    reclaim_checkpoint_db_space,
    run_checkpoint_maintenance,
)


def _make_checkpoint_db(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
        "parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, metadata BLOB, "
        "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
    )
    cur.execute(
        "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
        "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
    )
    return conn


def _add_checkpoint(conn, thread_id, checkpoint_id, *, checkpoint_ns="", blob=b"x") -> None:
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
        "type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (thread_id, checkpoint_ns, checkpoint_id, None, "msgpack", blob, b""),
    )
    conn.execute(
        "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (thread_id, checkpoint_ns, checkpoint_id, "task", 0, "channel", "msgpack", b"v"),
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


@pytest.mark.asyncio
async def test_build_live_state_snapshot_merges_chat_history(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    append_chat_message(conversation_id="conv-1", role="user", content="hello", agent_key="agent-1")
    append_chat_message(conversation_id="conv-1", role="assistant", content="world", agent_key="agent-1")

    class _Snapshot:
        values = {"decision": {"action": "respond"}, "conversation_id": "conv-1"}
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

    snapshot = await build_live_state_snapshot(thread_id="t1", keys=["chat_history"], include_all=False)
    assert snapshot["chat_history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


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


def test_delete_conversation_state_removes_checkpoint_and_chat_history(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, metadata TEXT)")
    cur.execute("INSERT INTO checkpoints (thread_id, checkpoint_ns, metadata) VALUES (?, ?, ?)", (
        "thread-1",
        "",
        '{"conversation_id":"conv-1"}',
    ))
    cur.execute("INSERT INTO checkpoints (thread_id, checkpoint_ns, metadata) VALUES (?, ?, ?)", (
        "thread-2",
        "",
        '{"conversation_id":"conv-2"}',
    ))
    conn.commit()
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    append_chat_message(conversation_id="conv-1", role="user", content="hello")
    append_chat_message(conversation_id="conv-2", role="user", content="world")

    assert delete_conversation_state("conv-1") is True

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("thread-1",))
    assert cur.fetchone() == (0,)
    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("thread-2",))
    assert cur.fetchone() == (1,)
    cur.execute("SELECT COUNT(*) FROM chat_history WHERE conversation_id = ?", ("conv-1",))
    assert cur.fetchone() == (0,)
    cur.execute("SELECT COUNT(*) FROM chat_history WHERE conversation_id = ?", ("conv-2",))
    assert cur.fetchone() == (1,)
    cur.execute("SELECT COUNT(*) FROM conversations WHERE conversation_id = ?", ("conv-1",))
    assert cur.fetchone() == (0,)
    conn.close()


def test_load_conversation_metadata_returns_persisted_worked_seconds(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    append_chat_message(
        conversation_id="conv-1",
        role="user",
        content="hello",
        agent_key="agent-1",
    )

    metadata = load_conversation_metadata("conv-1")

    assert metadata["conversation_id"] == "conv-1"
    assert metadata["agent_key"] == "agent-1"
    assert metadata["worked_seconds"] == 0
    assert get_conversation_metadata("conv-1") == metadata


def test_load_conversation_snapshot_uses_persisted_chat_history(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    append_chat_message(conversation_id="conv-1", role="user", content="hello", agent_key="agent-1")
    append_chat_message(
        conversation_id="conv-1",
        role="assistant",
        content="world",
        agent_key="agent-1",
    )

    snapshot = load_conversation_snapshot("conv-1")

    assert snapshot["conversation_id"] == "conv-1"
    assert snapshot["agent_key"] == "agent-1"
    assert snapshot["last_run_status"] is None
    assert snapshot["chat_history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def test_chat_history_service_migrates_conversations_without_worked_seconds(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            agent_key TEXT,
            worked_seconds INTEGER NOT NULL DEFAULT 0,
            last_thread_id TEXT,
            last_run_id TEXT,
            last_run_status TEXT,
            created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
            updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        )
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    append_chat_message(conversation_id="conv-legacy", role="user", content="hello")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(conversations)")
    columns = [row[1] for row in cur.fetchall() if len(row) > 1]
    assert "worked_seconds" not in columns
    cur.execute("SELECT conversation_id, agent_key FROM conversations WHERE conversation_id = ?", ("conv-legacy",))
    assert cur.fetchone() == ("conv-legacy", None)
    conn.close()


def test_prune_thread_checkpoints_keeps_newest_and_drops_orphan_writes(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = _make_checkpoint_db(db_path)
    # checkpoint_id sorts ascending; cp-3 is newest for thread-1.
    for cid in ("cp-1", "cp-2", "cp-3"):
        _add_checkpoint(conn, "thread-1", cid)
    _add_checkpoint(conn, "thread-2", "cp-9")
    conn.commit()
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    deleted = prune_thread_checkpoints("thread-1", keep_last=1)
    assert deleted == 2

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?", ("thread-1",))
    assert cur.fetchall() == [("cp-3",)]
    # writes for the deleted checkpoints are gone; the surviving one is kept.
    cur.execute("SELECT checkpoint_id FROM writes WHERE thread_id = ?", ("thread-1",))
    assert cur.fetchall() == [("cp-3",)]
    # other threads are untouched.
    cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("thread-2",))
    assert cur.fetchone() == (1,)
    conn.close()


def test_prune_thread_checkpoints_respects_keep_last(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = _make_checkpoint_db(db_path)
    for cid in ("cp-1", "cp-2", "cp-3", "cp-4"):
        _add_checkpoint(conn, "thread-1", cid)
    conn.commit()
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    assert prune_thread_checkpoints("thread-1", keep_last=2) == 2

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id", ("thread-1",))
    assert cur.fetchall() == [("cp-3",), ("cp-4",)]
    conn.close()


def test_reclaim_checkpoint_db_space_converts_to_incremental_and_shrinks(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = _make_checkpoint_db(db_path)
    # Bulk-insert then delete to create free pages on a NONE auto_vacuum DB.
    for i in range(2000):
        _add_checkpoint(conn, "thread-1", f"cp-{i:05d}", blob=b"x" * 1024)
    conn.commit()
    conn.execute("DELETE FROM checkpoints")
    conn.execute("DELETE FROM writes")
    conn.commit()
    assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 0
    assert conn.execute("PRAGMA freelist_count").fetchone()[0] > 0
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))
    size_before = db_path.stat().st_size

    assert reclaim_checkpoint_db_space() is True

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # INCREMENTAL
    assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0
    conn.close()
    assert db_path.stat().st_size < size_before


def test_run_checkpoint_maintenance_prunes_and_reclaims(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    conn = _make_checkpoint_db(db_path)
    for cid in ("cp-1", "cp-2", "cp-3"):
        _add_checkpoint(conn, "thread-1", cid)
    conn.commit()
    conn.close()

    monkeypatch.setenv("SOLIDCUE_CHECKPOINT_DB_PATH", str(db_path))

    summary = run_checkpoint_maintenance("thread-1", keep_last=1)
    assert summary["thread_id"] == "thread-1"
    assert summary["pruned"] == 2
    assert summary["reclaimed"] is True

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", ("thread-1",)).fetchone() == (1,)
    conn.close()


@pytest.mark.asyncio
async def test_is_conversation_resumable_uses_latest_thread(monkeypatch) -> None:
    monkeypatch.setattr(
        "solidcue.services.state_snapshot_service.get_latest_thread_id_for_conversation",
        lambda conversation_id: "thread-123" if conversation_id == "conv-1" else None,
    )

    async def _fake_is_thread_resumable(thread_id: str) -> dict[str, object]:
      return {"resumable": True, "next_nodes": ["execution"]} if thread_id == "thread-123" else {"resumable": False, "next_nodes": []}

    monkeypatch.setattr(
        "solidcue.services.state_snapshot_service.is_thread_resumable",
        _fake_is_thread_resumable,
    )

    payload = await is_conversation_resumable("conv-1")

    assert payload == {
        "resumable": True,
        "next_nodes": ["execution"],
        "thread_id": "thread-123",
    }
