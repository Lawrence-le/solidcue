"""Tests for graph_system's planning_mode_node and the runtime cache gate."""

import importlib
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# planning_mode_node — the focused LLM classifier
# ---------------------------------------------------------------------------


def _node_module():
    return importlib.import_module(
        "solidcue.core.graph_system.nodes.planning_mode_node"
    )


def _provider_yielding(text: str) -> MagicMock:
    async def _stream(messages, **_):
        yield text

    fake = MagicMock()
    fake.async_stream_generate = _stream
    return fake


@pytest.mark.asyncio
async def test_planning_mode_static_from_llm(monkeypatch):
    pm = _node_module()
    monkeypatch.setattr(
        pm, "_get_workspace_provider", lambda: _provider_yielding('{"planning_mode": "static"}')
    )

    result = await pm.planning_mode_node(
        {"agent_spec": {"name": "Resume", "description": "fixed doc pipeline"}}
    )
    assert result["agent_spec"]["planning_mode"] == "static"


@pytest.mark.asyncio
async def test_planning_mode_dynamic_from_llm(monkeypatch):
    pm = _node_module()
    monkeypatch.setattr(
        pm, "_get_workspace_provider", lambda: _provider_yielding('{"planning_mode": "dynamic"}')
    )

    result = await pm.planning_mode_node(
        {"agent_spec": {"name": "Weather", "description": "varies per request"}}
    )
    assert result["agent_spec"]["planning_mode"] == "dynamic"


@pytest.mark.asyncio
async def test_planning_mode_garbage_defaults_dynamic(monkeypatch):
    """Unparseable / invalid model output falls back to the safe default."""
    pm = _node_module()
    monkeypatch.setattr(
        pm, "_get_workspace_provider", lambda: _provider_yielding("not json at all")
    )

    result = await pm.planning_mode_node({"agent_spec": {"name": "X", "description": "d"}})
    assert result["agent_spec"]["planning_mode"] == "dynamic"


@pytest.mark.asyncio
async def test_planning_mode_no_provider_defaults_dynamic(monkeypatch):
    pm = _node_module()
    monkeypatch.setattr(pm, "_get_workspace_provider", lambda: None)

    result = await pm.planning_mode_node({"agent_spec": {"name": "X", "description": "d"}})
    assert result["agent_spec"]["planning_mode"] == "dynamic"


@pytest.mark.asyncio
async def test_planning_mode_respects_explicit_choice(monkeypatch):
    """An explicit upstream choice is kept; the provider must not be called."""
    pm = _node_module()

    def _boom():
        raise AssertionError("provider should not be called")

    monkeypatch.setattr(pm, "_get_workspace_provider", _boom)

    result = await pm.planning_mode_node(
        {"agent_spec": {"name": "X", "description": "d", "planning_mode": "static"}}
    )
    assert result["agent_spec"]["planning_mode"] == "static"


# ---------------------------------------------------------------------------
# Runtime gate — the mode actually controls whether the task plan caches
# ---------------------------------------------------------------------------


def test_dynamic_agent_plan_not_cacheable(monkeypatch):
    pn = importlib.import_module("solidcue.core.graph_agent.nodes.planning_node")
    from solidcue.agent_configs.schema import AgentConfig, PlanningPolicy, ProviderConfig

    p = ProviderConfig(type="o", api_key_env="E", model="m")

    def _cfg(mode):
        return AgentConfig(
            agent_key="k", name="n", provider=p, planning=PlanningPolicy(mode=mode)
        )

    monkeypatch.setattr(pn, "load_agent", lambda key: _cfg("dynamic"))
    assert pn._plan_is_cacheable("k") is False

    monkeypatch.setattr(pn, "load_agent", lambda key: _cfg("static"))
    assert pn._plan_is_cacheable("k") is True


def test_plan_cacheable_false_on_load_error(monkeypatch):
    pn = importlib.import_module("solidcue.core.graph_agent.nodes.planning_node")

    def _boom(key):
        raise RuntimeError("no such agent")

    monkeypatch.setattr(pn, "load_agent", _boom)
    assert pn._plan_is_cacheable("k") is False
    assert pn._plan_is_cacheable("") is False
