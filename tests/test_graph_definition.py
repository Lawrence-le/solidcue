"""Tests for graph_definition: standalone writer subgraphs and node units."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

generate_module = importlib.import_module("solidcue.core.graph_definition.nodes.generate_node")

from solidcue.core.graph_definition.builder import (
    build_definition_graph,
    build_persona_graph,
    build_skill_graph,
    build_tools_graph,
)
from solidcue.core.graph_definition.nodes.load_contract_node import load_contract_node
from solidcue.core.graph_definition.nodes.write_node import write_node


# ---------------------------------------------------------------------------
# load_contract_node
# ---------------------------------------------------------------------------


def test_load_contract_persona_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "solidcue.core.graph_definition.nodes.load_contract_node.SKILLS_ROOT_DIR", tmp_path
    )
    (tmp_path / "create-persona.md").write_text("# Create Persona\nContract text.")
    result = load_contract_node({"definition_target": "persona"})
    assert result["contract_skill"] == "# Create Persona\nContract text."


def test_load_contract_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "solidcue.core.graph_definition.nodes.load_contract_node.SKILLS_ROOT_DIR", tmp_path
    )
    result = load_contract_node({"definition_target": "skill"})
    assert result["contract_skill"] == ""


def test_load_contract_no_target_returns_empty():
    result = load_contract_node({})
    assert result["contract_skill"] == ""


# ---------------------------------------------------------------------------
# write_node
# ---------------------------------------------------------------------------


def test_write_node_persona(tmp_path, monkeypatch):
    import solidcue.agent_configs.loader as loader_mod

    def fake_save_persona(agent_key, content=None, *, overwrite=False):
        path = tmp_path / agent_key / "PERSONA.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content or "default")
        return path

    monkeypatch.setattr(loader_mod, "save_agent_persona", fake_save_persona)

    result = write_node({
        "definition_target": "persona",
        "agent_key": "test_agent",
        "definition_content": "# My Persona",
        "overwrite": False,
    })
    assert result["definition_path"].endswith("PERSONA.md")


def test_write_node_unknown_target_returns_empty():
    result = write_node({"definition_target": "unknown", "agent_key": "x", "definition_content": ""})
    assert result["definition_path"] == ""


def test_write_node_missing_agent_key_returns_empty():
    result = write_node({"definition_target": "persona", "agent_key": "", "definition_content": "x"})
    assert result["definition_path"] == ""


# ---------------------------------------------------------------------------
# generate_node (async)
# ---------------------------------------------------------------------------


async def _fake_async_stream(messages, **_kwargs):
    yield "# Generated Content"


@pytest.mark.asyncio
async def test_generate_node_with_provider(monkeypatch):
    fake_provider = MagicMock()
    fake_provider.async_stream_generate = _fake_async_stream
    monkeypatch.setattr(generate_module, "_get_workspace_provider", lambda: fake_provider)

    result = await generate_module.generate_node({
        "definition_target": "persona",
        "contract_skill": "# Contract",
        "agent_spec": {"name": "Test", "agent_key": "test_agent"},
    })
    assert "# Generated Content" in result["definition_content"]


async def _fenced_async_stream(messages, **_kwargs):
    yield "```markdown\n# Persona\n\nrole text\n```"


@pytest.mark.asyncio
async def test_generate_node_strips_code_fence(monkeypatch):
    """The model often wraps output in ```markdown … ``` — it must not reach disk."""
    fake_provider = MagicMock()
    fake_provider.async_stream_generate = _fenced_async_stream
    monkeypatch.setattr(generate_module, "_get_workspace_provider", lambda: fake_provider)

    result = await generate_module.generate_node({
        "definition_target": "persona",
        "contract_skill": "# Contract",
        "agent_spec": {"name": "Test", "agent_key": "test_agent"},
    })
    content = result["definition_content"]
    assert not content.startswith("```")
    assert "```" not in content
    assert content.startswith("# Persona")


def test_strip_code_fence_unit():
    f = generate_module._strip_code_fence
    assert f("```markdown\n# A\n```") == "# A"
    assert f("```\n# B\nx\n```") == "# B\nx"
    assert f("# C\nplain") == "# C\nplain"


@pytest.mark.asyncio
async def test_generate_node_no_provider_returns_empty(monkeypatch):
    monkeypatch.setattr(generate_module, "_get_workspace_provider", lambda: None)

    result = await generate_module.generate_node({
        "definition_target": "persona",
        "contract_skill": "# Contract",
        "agent_spec": {},
    })
    assert result["definition_content"] == ""


# ---------------------------------------------------------------------------
# build_definition_graph — standalone writer (persona / skill / tools)
# ---------------------------------------------------------------------------


async def _run_definition_graph(target: str, agent_key: str, tmp_path: Path, monkeypatch) -> dict:
    """Helper: stub provider + file IO, then run the subgraph end-to-end."""
    import solidcue.agent_configs.loader as loader_mod
    import solidcue.core.graph_definition.nodes.generate_node as gmod
    import solidcue.core.graph_definition.nodes.load_contract_node as lcmod

    generated_content = f"# Generated {target.title()} Content"

    async def _stream(messages, **_):
        yield generated_content

    fake_provider = MagicMock()
    fake_provider.async_stream_generate = _stream
    monkeypatch.setattr(gmod, "_get_workspace_provider", lambda: fake_provider)

    monkeypatch.setattr(lcmod, "SKILLS_ROOT_DIR", tmp_path)
    (tmp_path / f"create-{target}.md").write_text(f"# Contract for {target}")

    written: dict[str, str] = {}

    def _make_saver(label: str):
        def _save(ak, content=None, *, overwrite=False):
            path = tmp_path / ak / f"{label}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content or "")
            written[label] = content or ""
            return path
        return _save

    # Patch at the loader module level (write_node uses _loader.save_agent_*)
    monkeypatch.setattr(loader_mod, "save_agent_persona", _make_saver("PERSONA"))
    monkeypatch.setattr(loader_mod, "save_agent_skill",   _make_saver("SKILL"))
    monkeypatch.setattr(loader_mod, "save_agent_tools",   _make_saver("TOOLS"))

    graph = build_definition_graph(target)
    async for _ in graph.astream(
        {
            "definition_target": target,
            "agent_key": agent_key,
            "agent_spec": {"name": "X", "agent_key": agent_key},
            "overwrite": True,
        },
        stream_mode=["updates"],
    ):
        pass

    return {"written": written}


@pytest.mark.asyncio
async def test_persona_graph_standalone(tmp_path, monkeypatch):
    result = await _run_definition_graph("persona", "my_agent", tmp_path, monkeypatch)
    assert "PERSONA" in result["written"]
    assert "Generated Persona" in result["written"]["PERSONA"]


@pytest.mark.asyncio
async def test_skill_graph_standalone(tmp_path, monkeypatch):
    result = await _run_definition_graph("skill", "my_agent", tmp_path, monkeypatch)
    assert "SKILL" in result["written"]
    assert "Generated Skill" in result["written"]["SKILL"]


@pytest.mark.asyncio
async def test_tools_graph_standalone(tmp_path, monkeypatch):
    result = await _run_definition_graph("tools", "my_agent", tmp_path, monkeypatch)
    assert "TOOLS" in result["written"]
    assert "Generated Tools" in result["written"]["TOOLS"]


# ---------------------------------------------------------------------------
# Factory functions return distinct compiled graphs
# ---------------------------------------------------------------------------


def test_factory_functions_compile():
    g_persona = build_persona_graph()
    g_skill = build_skill_graph()
    g_tools = build_tools_graph()
    assert g_persona is not None
    assert g_skill is not None
    assert g_tools is not None
    assert g_persona is not g_skill
