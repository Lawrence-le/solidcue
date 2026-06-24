"""Router behavior for the create_agent intent.

The router answers conversationally — it does NOT hard-stop on a spec form. The
actual agent-creation machinery (graph_system / graph_definition) is exercised by
test_graph_system_create_agent.py; this file guards the chat-facing behavior.
"""

from __future__ import annotations

import importlib

import pytest

from tests.test_graph_system_create_agent import _wire_stubs

intent_router_module = importlib.import_module(
    "solidcue.core.graph_router.nodes.intent_router_node"
)

_CREATE_AGENT_JSON = (
    '{"assistant_draft":"Sure, I can help you create a new agent. What should it '
    'do, and what would you like to call it?",'
    '"router_intent":"create_agent","router_next":"final_output",'
    '"target_agent_key":"","route_reason":"User asked to create an agent",'
    '"plan":[],"handoff":{},"target_artifacts_source":[]}'
)


class _RouterProvider:
    model = "router-stream-model"

    def generate(self, _messages, **_kwargs):
        return _CREATE_AGENT_JSON


def _build_router_graph():
    from langgraph.checkpoint.memory import InMemorySaver
    from solidcue.core.graph_router.builder import _compile_graph
    return _compile_graph(InMemorySaver())


def _stub_router_provider(monkeypatch):
    monkeypatch.setattr(
        intent_router_module,
        "resolve_router_provider",
        lambda _thread_id: _RouterProvider(),
    )


_READY_JSON = (
    '{"assistant_draft":"Great, I\'ll create the weather_assistant now.",'
    '"router_intent":"create_agent","router_next":"final_output",'
    '"target_agent_key":"","route_reason":"User confirmed agent details",'
    '"plan":[],"handoff":{},"target_artifacts_source":[],'
    '"agent_ready":true,'
    '"agent_spec":{"name":"Weather Assistant","agent_key":"weather_assistant",'
    '"description":"Provides current weather and forecasts","selected_tools":[]}}'
)


class _ReadyRouterProvider:
    model = "router-stream-model"

    def generate(self, _messages, **_kwargs):
        return _READY_JSON


@pytest.mark.asyncio
async def test_intent_router_seeds_spec_when_ready(monkeypatch):
    monkeypatch.setattr(
        intent_router_module,
        "resolve_router_provider",
        lambda _thread_id: _ReadyRouterProvider(),
    )

    result = await intent_router_module.intent_router_node(
        {"thread_id": "ready-1", "user_input": "provide current weather and forecast", "chat_history": []}
    )

    assert result["router_intent"] == "create_agent"
    assert result["system_intent"] == "create_agent"
    spec = result["agent_spec"]
    assert spec["agent_key"] == "weather_assistant"
    assert spec["name"] == "Weather Assistant"


@pytest.mark.asyncio
async def test_router_create_agent_completes_when_ready(tmp_path, monkeypatch):
    """When the conversation is ready, the router actually builds the agent —
    inheriting the workspace provider (no form)."""
    written = _wire_stubs(tmp_path, monkeypatch, agent_key="weather_assistant")
    monkeypatch.setattr(
        intent_router_module,
        "resolve_router_provider",
        lambda _thread_id: _ReadyRouterProvider(),
    )
    # Inherit a workspace provider so collect_spec validates without a form.
    cs_mod = importlib.import_module("solidcue.core.graph_system.nodes.collect_spec_node")
    monkeypatch.setattr(
        cs_mod,
        "_workspace_provider_defaults",
        lambda: {
            "decision_provider_type": "anthropic", "decision_base_url": None,
            "decision_api_key": "sk-ws", "decision_model": "m", "decision_temperature": 0.2,
            "lite_provider_type": "anthropic", "lite_base_url": None,
            "lite_api_key": "sk-ws", "lite_model": "m", "lite_temperature": 0.2,
            "reviewer_provider_type": "anthropic", "reviewer_base_url": None,
            "reviewer_api_key": "sk-ws", "reviewer_model": "m", "reviewer_temperature": 0.2,
            "selected_tools": [],
        },
    )

    graph = _build_router_graph()
    result = await graph.ainvoke(
        {"thread_id": "r-done", "user_input": "provide current weather and forecast", "metadata": {}},
        config={"configurable": {"thread_id": "r-done"}},
    )

    assert "__interrupt__" not in result
    assert result.get("created_agent_key") == "weather_assistant"
    assert {"PERSONA", "SKILL", "TOOLS"} <= set(written)


@pytest.mark.asyncio
async def test_router_create_agent_answers_conversationally(monkeypatch):
    _stub_router_provider(monkeypatch)

    graph = _build_router_graph()
    result = await graph.ainvoke(
        {"thread_id": "r-ca-1", "user_input": "can you help me with agent creation", "metadata": {}},
        config={"configurable": {"thread_id": "r-ca-1"}},
    )

    # The router replies and finishes the turn — no form interrupt, no dead-end.
    assert "__interrupt__" not in result
    assert result["router_intent"] == "create_agent"
    assert "help with that" in result["final_response"].lower()
    assert not result.get("created_agent_key")
