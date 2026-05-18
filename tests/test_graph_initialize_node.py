import importlib

from solidcue.core.graph_node.initialize_node import initialize_node

initialize_module = importlib.import_module("solidcue.core.graph_node.initialize_node")


_DISALLOWED_INITIALIZE_WRITES = (
    "decision",
    "synthesis_draft",
    "final_response",
    "retry_reason",
    "draft_output",
    "finalization_reason",
    "router_origin",
    "latest_output",
    "execution_result",
)


def test_initialize_sets_phase_default() -> None:
    result = initialize_node({})

    assert result["phase"] == "source"


def test_initialize_preserves_explicit_phase() -> None:
    result = initialize_node({"phase": "artifact"})

    assert result["phase"] == "artifact"

def test_initialize_initializes_attempt_counters() -> None:
    result = initialize_node({})

    assert result["source_attempt"] == 0
    assert result["artifact_attempt"] == 0
    assert result["synthesis_attempt"] == 0


def test_initialize_increments_attempt_counters_from_state() -> None:
    result = initialize_node(
        {
            "source_attempt": 2,
            "artifact_attempt": 1,
            "synthesis_attempt": 0,
        }
    )

    assert result["source_attempt"] == 2
    assert result["artifact_attempt"] == 1
    assert result["synthesis_attempt"] == 0


def test_initialize_sets_metadata_with_current_time() -> None:
    result = initialize_node({})

    assert "metadata" in result
    assert "current_time" in result["metadata"]
    assert "current_date" in result["metadata"]
    assert "current_time_utc" in result["metadata"]
    assert "timezone" in result["metadata"]
    assert "location" in result["metadata"]
    assert "UTC" in result["metadata"]["current_time_utc"]


def test_initialize_preserves_existing_metadata() -> None:
    result = initialize_node({"metadata": {"user_id": "123"}})

    assert result["metadata"]["user_id"] == "123"
    assert "current_time_utc" in result["metadata"]


def test_initialize_uses_config_timezone_and_location() -> None:
    result = initialize_node(
        {
            "config": {
                "timezone": "Asia/Singapore",
                "location": "Singapore",
            }
        }
    )

    assert result["metadata"]["timezone"] == "Asia/Singapore"
    assert result["metadata"]["location"] == "Singapore"


def test_initialize_uses_env_default_timezone_when_config_missing(monkeypatch) -> None:
    monkeypatch.setenv("SOLIDCUE_DEFAULT_TIMEZONE", "Asia/Singapore")

    result = initialize_node({})

    assert result["metadata"]["timezone"] == "Asia/Singapore"


def test_initialize_falls_back_max_retries_to_three() -> None:
    result = initialize_node({"max_retries": "bad"})

    assert result["max_retries"] == 3


def test_initialize_omits_disallowed_artifact_and_synthesis_writes() -> None:
    result = initialize_node({})

    for key in _DISALLOWED_INITIALIZE_WRITES:
        assert key not in result, f"initialize_node must not write '{key}'"


def test_initialize_does_not_preload_hhem_by_default(monkeypatch) -> None:
    called = {"value": False}
    monkeypatch.delenv("SOLIDCUE_HHEM_PRELOAD", raising=False)
    monkeypatch.setattr(initialize_module, "load_hhem_model", lambda: called.update(value=True))

    initialize_node({})

    assert called["value"] is False


def test_initialize_preloads_hhem_when_enabled(monkeypatch) -> None:
    called = {"value": False}
    monkeypatch.setenv("SOLIDCUE_HHEM_PRELOAD", "true")
    monkeypatch.setattr(initialize_module, "load_hhem_model", lambda: called.update(value=True))

    initialize_node({})

    assert called["value"] is True
