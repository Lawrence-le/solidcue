"""Tests for the MCP playbook registry (fetch + cache + resolution)."""

import importlib

import pytest


def _registry():
    return importlib.import_module("solidcue.tools.playbook_registry")


def _mcp_tool(tool_key: str, server_key: str):
    return type("Tool", (), {
        "type": "mcp", "enabled": True, "tool_key": tool_key,
        "mcp": type("M", (), {"server_key": server_key})(),
    })()


def _client_with(resources, texts):
    class _Client:
        def __init__(self, server): ...
        async def list_resources(self):
            return resources
        async def read_resource(self, uri):
            return texts[uri]
    return _Client


@pytest.mark.asyncio
async def test_refresh_caches_playbook_from_server(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()

    monkeypatch.setattr(reg, "list_tools", lambda: [_mcp_tool("drive_download_file", "google_drive")])
    monkeypatch.setattr(reg, "load_mcp_server", lambda key: type("S", (), {"enabled": True})())
    monkeypatch.setattr(reg, "MCPClient", _client_with(
        [{"uri": "playbook://google-drive", "mime_type": "text/markdown"}],
        {"playbook://google-drive": "# Playbook\nsequence rules"},
    ))

    await reg.refresh_all()
    assert reg.get_server_playbook("google_drive") == "# Playbook\nsequence rules"


@pytest.mark.asyncio
async def test_resolve_playbook_via_tool(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()
    reg._PLAYBOOK_CACHE["google_drive"] = "# Playbook"

    tool_config = type("T", (), {"mcp": type("M", (), {"server_key": "google_drive"})()})()
    monkeypatch.setattr(reg, "load_tool", lambda key: tool_config)

    assert reg.get_playbook_for_tool("drive_upload_file") == "# Playbook"


@pytest.mark.asyncio
async def test_server_without_playbook_resource_is_skipped(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()

    monkeypatch.setattr(reg, "list_tools", lambda: [_mcp_tool("get_weather_forecast", "open_meteo")])
    monkeypatch.setattr(reg, "load_mcp_server", lambda key: type("S", (), {"enabled": True})())
    # Server exposes a non-playbook resource only.
    monkeypatch.setattr(reg, "MCPClient", _client_with(
        [{"uri": "file://readme.txt"}], {},
    ))

    await reg.refresh_all()
    assert reg.get_server_playbook("open_meteo") is None


@pytest.mark.asyncio
async def test_unreachable_server_is_swallowed(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()

    monkeypatch.setattr(reg, "list_tools", lambda: [_mcp_tool("drive_download_file", "google_drive")])
    monkeypatch.setattr(reg, "load_mcp_server", lambda key: type("S", (), {"enabled": True})())

    class _DeadClient:
        def __init__(self, server): ...
        async def list_resources(self):
            raise RuntimeError("server down")

    monkeypatch.setattr(reg, "MCPClient", _DeadClient)

    await reg.refresh_all()  # must not raise
    assert reg.get_server_playbook("google_drive") is None


@pytest.mark.asyncio
async def test_warmed_once(monkeypatch):
    reg = _registry()
    reg.reset_for_tests()
    calls = {"n": 0}

    def _count():
        calls["n"] += 1
        return []

    monkeypatch.setattr(reg, "list_tools", _count)

    await reg.ensure_playbooks_warmed()
    await reg.ensure_playbooks_warmed()
    assert calls["n"] == 1  # second call is a no-op
