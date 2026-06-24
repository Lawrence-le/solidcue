"""Tests for multi-item synthesis: structured data, aggregation, and labels.

Covers the three generic fixes:
1. structured (non-text) handoff values are serialized, not dropped
2. aggregate synthesis reads every item, not just one
3. per-item labels are captured at execution and surfaced by synthesis
"""

import importlib

sn = importlib.import_module("solidcue.core.graph_agent.nodes.synthesis_node")
en = importlib.import_module("solidcue.core.graph_agent.nodes.execution_node")


# --- Fix 1: structured data is readable --------------------------------------


def test_stringify_prefers_text_field():
    assert sn._stringify_handoff_value({"content": "hello"}) == "hello"


def test_stringify_serializes_structured_dict():
    # No text/content/body field -> serialize whole object instead of dropping.
    out = sn._stringify_handoff_value({"temperature": "26.6C", "humidity": "38%"})
    assert "temperature" in out and "26.6C" in out


def test_stringify_plain_string_and_list():
    assert sn._stringify_handoff_value("  hi  ") == "hi"
    assert "a" in sn._stringify_handoff_value(["a", "b"])


# --- Fix 2: aggregate across all items ---------------------------------------


def test_is_aggregate_task():
    assert sn._is_aggregate_task({"context": {"scope": "all"}}) is True
    assert sn._is_aggregate_task({"context": {"scope": "aggregate"}}) is True
    assert sn._is_aggregate_task({"context": {"item_key": "item_1"}}) is False
    assert sn._is_aggregate_task({}) is False


def test_aggregate_source_includes_every_item_with_labels():
    handoff = {
        "weather_data_retrieved::item_1": {"temperature": "21.2C"},
        "weather_data_retrieved::item_2": {"temperature": "36.3C"},
        "weather_data_retrieved::item_3": {"temperature": "26.6C"},
        "item_label::item_1": "London",
        "item_label::item_2": "New York",
        "item_label::item_3": "Tokyo",
    }
    out = sn._build_aggregate_source(handoff)
    # all three items present, each under its label
    for label, temp in [("London", "21.2C"), ("New York", "36.3C"), ("Tokyo", "26.6C")]:
        assert label in out
        assert temp in out


def test_aggregate_source_falls_back_to_slot_key_without_label():
    handoff = {"data::item_1": {"v": "1"}, "data::item_2": {"v": "2"}}
    out = sn._build_aggregate_source(handoff)
    assert "item_1" in out and "item_2" in out


def test_build_source_routes_aggregate_task():
    state = {
        "handoff": {
            "data::item_1": {"v": "a"},
            "data::item_2": {"v": "b"},
            "item_label::item_1": "Alpha",
            "item_label::item_2": "Beta",
        },
        "task_plan": [{"id": "task_1", "type": "synthesis", "context": {"scope": "all"}}],
        "current_task": "task_1",
    }
    out = sn._build_source_from_handoff(state)
    assert "Alpha" in out and "Beta" in out


# --- Fix 3: per-item label capture -------------------------------------------


def test_write_handoff_records_label_from_target_artifacts_source():
    state = {
        "handoff": {},
        "task_plan": [{"id": "task_1", "requires": ["data_got"], "context": {"item_key": "item_1"}}],
        "current_task": "task_1",
        "target_artifacts_source": [
            {"item_key": "item_1", "source_ref": "https://example.com/job/1"},
        ],
    }
    result = en._write_handoff(state, {"success": True, "content": {"x": 1}})
    assert result["data_got::item_1"] == {"x": 1}
    assert result["item_label::item_1"] == "https://example.com/job/1"


def test_write_handoff_no_label_when_source_missing():
    # Free-text item not in target_artifacts_source -> no label written; synthesis
    # falls back to the slot key. Graceful, generic degradation.
    state = {
        "handoff": {},
        "task_plan": [{"id": "task_1", "requires": ["data_got"], "context": {"item_key": "item_1"}}],
        "current_task": "task_1",
        "target_artifacts_source": [],
    }
    result = en._write_handoff(state, {"success": True, "content": {"x": 1}})
    assert "item_label::item_1" not in result
