from __future__ import annotations

import importlib
from types import SimpleNamespace

from solidcue.core.graph_router.nodes.initialize_router_node import initialize_router_node
from solidcue.core.graph_router.prompts.router_prompt import build_router_messages

initialize_router_module = importlib.import_module(
    "solidcue.core.graph_router.nodes.initialize_router_node"
)


def test_initialize_router_node_sets_metadata_defaults(monkeypatch) -> None:
    monkeypatch.setenv("SOLIDCUE_DEFAULT_TIMEZONE", "Asia/Singapore")

    result = initialize_router_node({"config": {"location": "Singapore"}})

    metadata = result["metadata"]
    assert metadata["timezone"] == "Asia/Singapore"
    assert metadata["location"] == "Singapore"
    assert "current_time" in metadata
    assert "current_date" in metadata
    assert "current_time_utc" in metadata


def test_initialize_router_node_uses_profile_location_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        initialize_router_module,
        "load_user_profile",
        lambda: SimpleNamespace(location="Singapore"),
    )

    result = initialize_router_node({"metadata": {}, "config": {}})

    assert result["metadata"]["location"] == "Singapore"


def test_build_router_messages_includes_metadata() -> None:
    messages = build_router_messages(
        user_input="hello",
        chat_history=[],
        available_agents=[],
        metadata={"timezone": "Asia/Singapore", "location": "Singapore"},
    )

    assert "METADATA:" in messages[1]["content"]
    assert '"timezone": "Asia/Singapore"' in messages[1]["content"]
    assert '"location": "Singapore"' in messages[1]["content"]
