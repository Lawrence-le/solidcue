"""Tests for the MCP tool-schema registry and decision-prompt tool scoping."""

import importlib

import pytest


# ---------------------------------------------------------------------------
# schema_registry — resolution + live-refresh + fallback
# ---------------------------------------------------------------------------


def _registry():
    return importlib.import_module("solidcue.tools.schema_registry")


def test_get_schema_falls_back_to_yaml(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()

    # Nothing cached -> resolves the YAML snapshot via load_tool.
    fake_tool = type("T", (), {"mcp": type("M", (), {"input_schema": {"required": ["lat"]}})()})()
    monkeypatch.setattr(reg, "load_tool", lambda key: fake_tool)

    assert reg.get_tool_input_schema("get_weather_forecast") == {"required": ["lat"]}


def test_get_schema_prefers_cache_over_yaml(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()
    reg._SCHEMA_CACHE["t1"] = {"required": ["fresh"]}

    def _boom(key):
        raise AssertionError("YAML should not be read when cache has the tool")

    monkeypatch.setattr(reg, "load_tool", _boom)
    assert reg.get_tool_input_schema("t1") == {"required": ["fresh"]}


@pytest.mark.asyncio
async def test_refresh_populates_cache_from_server(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()

    tool = type("Tool", (), {
        "type": "mcp", "enabled": True, "tool_key": "wx",
        "mcp": type("M", (), {"server_key": "open_meteo", "tool_name": "get_weather_forecast"})(),
    })()
    monkeypatch.setattr(reg, "list_tools", lambda: [tool])
    monkeypatch.setattr(reg, "load_mcp_server", lambda key: type("S", (), {"enabled": True})())

    class _Client:
        def __init__(self, server): ...
        async def list_tools(self):
            return [{"name": "get_weather_forecast", "input_schema": {"required": ["lat", "lon"]}}]

    monkeypatch.setattr(reg, "MCPClient", _Client)

    await reg.refresh_all()
    assert reg.get_tool_input_schema("wx") == {"required": ["lat", "lon"]}


@pytest.mark.asyncio
async def test_refresh_server_error_keeps_yaml_fallback(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()

    tool = type("Tool", (), {
        "type": "mcp", "enabled": True, "tool_key": "wx",
        "mcp": type("M", (), {"server_key": "open_meteo", "tool_name": "get_weather_forecast"})(),
    })()
    monkeypatch.setattr(reg, "list_tools", lambda: [tool])
    monkeypatch.setattr(reg, "load_mcp_server", lambda key: type("S", (), {"enabled": True})())

    class _DeadClient:
        def __init__(self, server): ...
        async def list_tools(self):
            raise RuntimeError("server down")

    monkeypatch.setattr(reg, "MCPClient", _DeadClient)
    # YAML fallback for the uncached tool.
    yaml_tool = type("T", (), {"mcp": type("M", (), {"input_schema": {"required": ["lat"]}})()})()
    monkeypatch.setattr(reg, "load_tool", lambda key: yaml_tool)

    await reg.refresh_all()
    assert "wx" not in reg._SCHEMA_CACHE
    assert reg.get_tool_input_schema("wx") == {"required": ["lat"]}


@pytest.mark.asyncio
async def test_warmed_once(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()
    calls = {"n": 0}

    def _count():
        calls["n"] += 1
        return []

    monkeypatch.setattr(reg, "list_tools", _count)

    await reg.ensure_schemas_warmed()
    await reg.ensure_schemas_warmed()
    assert calls["n"] == 1  # second call is a no-op


# ---------------------------------------------------------------------------
# decision prompt — tool scoping
# ---------------------------------------------------------------------------


def _scope():
    dp = importlib.import_module("solidcue.core.graph_agent.prompts.decision_prompt")
    return dp._scope_tools_for_task


def test_scope_to_planned_tool():
    scope = _scope()
    task = {"context": {"tool": "browser_navigate"}}
    assert scope(task, ["browser_navigate", "drive_upload_file"]) == ["browser_navigate"]


def test_scope_no_tool_returns_empty():
    scope = _scope()
    assert scope({"context": {"item_key": "item_1"}}, ["a", "b"]) == []


def test_scope_invalid_tool_falls_back_to_all():
    scope = _scope()
    task = {"context": {"tool": "not_an_allowed_tool"}}
    assert scope(task, ["a", "b"]) == ["a", "b"]
