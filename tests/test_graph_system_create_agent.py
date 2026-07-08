"""End-to-end tests for graph_system create_agent flow (Option B — pre-supplied spec)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Use importlib.import_module to get the actual module objects (not shadowed by __init__ exports).
gen_mod    = importlib.import_module("solidcue.core.graph_definition.nodes.generate_node")
lc_mod     = importlib.import_module("solidcue.core.graph_definition.nodes.load_contract_node")
verify_mod = importlib.import_module("solidcue.core.graph_system.nodes.verify_node")
wc_mod     = importlib.import_module("solidcue.core.graph_system.nodes.write_config_node")
loader_mod = importlib.import_module("solidcue.agent_configs.loader")

from solidcue.agent_configs.schema import AgentConfig, ProviderConfig
from solidcue.core.graph_system.nodes.collect_spec_node import (
    _validate_spec,
    collect_spec_node,
)

initialize_module = importlib.import_module("solidcue.core.graph_system.nodes.initialize_node")


_VALID_SPEC: dict[str, Any] = {
    "name": "Test Agent",
    "agent_key": "test_agent",
    "description": "A test agent",
    "decision_provider_type": "anthropic",
    "decision_base_url": None,
    "decision_api_key": "sk-test",
    "decision_model": "claude-haiku-4-5-20251001",
    "decision_temperature": 0.3,
    "lite_provider_type": "anthropic",
    "lite_base_url": None,
    "lite_api_key": "sk-test",
    "lite_model": "claude-haiku-4-5-20251001",
    "lite_temperature": 0.1,
    "reviewer_provider_type": "anthropic",
    "reviewer_base_url": None,
    "reviewer_api_key": "sk-test",
    "reviewer_model": "claude-haiku-4-5-20251001",
    "reviewer_temperature": 0.2,
    "selected_tools": [],
    # Definition substance now part of a "complete" spec — an explicit artifacts
    # answer + at least one key task, so collect_spec proceeds without interrupt.
    "produces_artifacts": False,
    "key_tasks": ["answer the test query"],
}

_FAKE_CONFIG = AgentConfig(
    agent_key="test_agent",
    name="Test Agent",
    description="A test agent",
    provider=ProviderConfig(type="anthropic", api_key_env="TEST_KEY", model="claude-haiku-4-5-20251001"),
)


# ---------------------------------------------------------------------------
# collect_spec_node unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_tools_keeps_valid_drops_invalid(monkeypatch):
    st_mod = importlib.import_module("solidcue.core.graph_system.nodes.select_tools_node")

    async def _stream(messages, **_):
        yield '{"selected_tools": ["get_weather_forecast", "made_up_tool"]}'

    fake = MagicMock()
    fake.async_stream_generate = _stream
    monkeypatch.setattr(st_mod, "_get_workspace_provider", lambda: fake)
    monkeypatch.setattr(
        st_mod,
        "_available_tools",
        lambda: [
            {"tool_key": "get_weather_forecast", "description": "w"},
            {"tool_key": "search_web", "description": "s"},
        ],
    )

    result = await st_mod.select_tools_node(
        {"agent_spec": {"name": "Weather", "agent_key": "wx", "description": "weather"}}
    )
    # Valid pick kept; invented one dropped.
    assert result["agent_spec"]["selected_tools"] == ["get_weather_forecast"]


@pytest.mark.asyncio
async def test_select_tools_respects_preselected(monkeypatch):
    st_mod = importlib.import_module("solidcue.core.graph_system.nodes.select_tools_node")
    monkeypatch.setattr(
        st_mod, "_available_tools",
        lambda: [{"tool_key": "search_web", "description": "s"}],
    )
    # Provider must not be called when tools are already chosen.
    def _boom():
        raise AssertionError("provider should not be called")
    monkeypatch.setattr(st_mod, "_get_workspace_provider", _boom)

    result = await st_mod.select_tools_node(
        {"agent_spec": {"name": "X", "agent_key": "x", "description": "d", "selected_tools": ["search_web"]}}
    )
    assert result["agent_spec"]["selected_tools"] == ["search_web"]


def test_validate_spec_helper():
    # A complete CreateAgentInput-shaped spec validates clean.
    assert _validate_spec(_VALID_SPEC) == []
    # An empty spec is missing the basic fields...
    empty = _validate_spec({})
    assert "name" in empty and "agent_key" in empty and "description" in empty
    # ...and a name-only spec is still missing the provider fields.
    partial = _validate_spec({"name": "X", "agent_key": "k", "description": "d"})
    assert "decision_provider_type" in partial
    assert "lite_model" in partial


def test_collect_spec_passes_valid_spec():
    # A complete spec does not interrupt — node returns directly.
    result = collect_spec_node({"agent_spec": _VALID_SPEC})
    assert result.get("system_next") != "final_output"
    assert result["created_agent_key"] == "test_agent"


# ---------------------------------------------------------------------------
# verify_node unit tests
# ---------------------------------------------------------------------------


def test_verify_node_no_agent_key():
    from solidcue.core.graph_system.nodes.verify_node import verify_node
    result = verify_node({})
    assert "Verification failed" in result["final_response"]


def test_verify_node_success(tmp_path, monkeypatch):
    from solidcue.core.graph_system.nodes.verify_node import verify_node

    monkeypatch.setattr(verify_mod, "load_agent", lambda key: MagicMock(agent_key=key))
    monkeypatch.setattr(verify_mod, "get_persona_path", lambda k: tmp_path / "PERSONA.md")
    monkeypatch.setattr(verify_mod, "get_skill_path",   lambda k: tmp_path / "SKILL.md")
    monkeypatch.setattr(verify_mod, "get_tools_path",   lambda k: tmp_path / "TOOLS.md")

    (tmp_path / "PERSONA.md").write_text("p")
    (tmp_path / "SKILL.md").write_text("s")
    (tmp_path / "TOOLS.md").write_text("t")

    result = verify_node({"created_agent_key": "test_agent"})
    assert "successfully" in result["final_response"]
    assert "test_agent" in result["final_response"]


def test_verify_node_missing_md(tmp_path, monkeypatch):
    from solidcue.core.graph_system.nodes.verify_node import verify_node

    monkeypatch.setattr(verify_mod, "load_agent", lambda key: MagicMock(agent_key=key))
    monkeypatch.setattr(verify_mod, "get_persona_path", lambda k: tmp_path / "PERSONA.md")
    monkeypatch.setattr(verify_mod, "get_skill_path",   lambda k: tmp_path / "SKILL.md")
    monkeypatch.setattr(verify_mod, "get_tools_path",   lambda k: tmp_path / "TOOLS.md")

    result = verify_node({"created_agent_key": "test_agent"})
    assert "issues" in result["final_response"]


# ---------------------------------------------------------------------------
# write_config_node unit tests
# ---------------------------------------------------------------------------


def test_write_config_node_success(monkeypatch):
    from solidcue.core.graph_system.nodes.write_config_node import write_config_node

    monkeypatch.setattr(wc_mod, "write_agent_config", lambda _: (_FAKE_CONFIG, "/tmp/test_agent.yaml"))

    result = write_config_node({"agent_spec": _VALID_SPEC})
    assert result["created_agent_key"] == "test_agent"
    assert result["created_config_path"] == "/tmp/test_agent.yaml"


def test_write_config_node_scrubs_api_keys(monkeypatch):
    from solidcue.core.graph_system.nodes.write_config_node import write_config_node

    monkeypatch.setattr(wc_mod, "write_agent_config", lambda _: (_FAKE_CONFIG, "/tmp/test_agent.yaml"))

    result = write_config_node({"agent_spec": _VALID_SPEC})
    scrubbed = result.get("agent_spec") or {}
    # Raw secrets must not survive into checkpointed state.
    assert not any(k.endswith("_api_key") for k in scrubbed)
    # Non-secret fields remain.
    assert scrubbed["agent_key"] == "test_agent"


def test_write_config_node_exception_routes_to_final_output(monkeypatch):
    from solidcue.core.graph_system.nodes.write_config_node import write_config_node

    def _boom(_):
        raise ValueError("kaboom")

    monkeypatch.setattr(wc_mod, "write_agent_config", _boom)

    result = write_config_node({"agent_spec": _VALID_SPEC})
    assert result.get("system_next") == "final_output"
    assert "Failed to write agent config" in result["final_response"]


def test_create_agent_input_coerces_string_list_fields():
    """The router LLM often emits list fields as a plain string. Coerce rather
    than raising a ValidationError that aborts agent creation."""
    from solidcue.services.agent_service import CreateAgentInput

    spec = dict(_VALID_SPEC)
    spec["selected_tools"] = "search_web, drive_upload_file"
    spec["key_tasks"] = "Search the web, extract facts, write summary"
    spec["examples"] = "none"  # malformed → dropped, not fatal

    parsed = CreateAgentInput(**spec)
    assert parsed.selected_tools == ["search_web", "drive_upload_file"]
    assert parsed.key_tasks == ["Search the web", "extract facts", "write summary"]
    assert parsed.examples == []


def test_write_config_node_survives_string_key_tasks(monkeypatch):
    """A string key_tasks used to crash CreateAgentInput → no YAML written; now
    it coerces and the config is built."""
    from solidcue.core.graph_system.nodes.write_config_node import write_config_node

    monkeypatch.setattr(wc_mod, "write_agent_config", lambda _: (_FAKE_CONFIG, "/tmp/test_agent.yaml"))
    spec = dict(_VALID_SPEC)
    spec["key_tasks"] = "one task, another task"

    result = write_config_node({"agent_spec": spec})
    assert result["created_agent_key"] == "test_agent"
    assert result.get("system_next") != "final_output"


def test_route_after_write_config_skips_verify_on_failure():
    """A failed write_config must not fall through to verify (which would report
    'created with issues' over a half-written agent)."""
    from solidcue.core.graph_system.builder import _route_after_write_config

    # No created_config_path (failure) → skip verify; present (success) → verify.
    assert _route_after_write_config({"created_agent_key": "x"}) == "final_output"
    assert _route_after_write_config({"created_config_path": "/tmp/x.yaml"}) == "verify"


# ---------------------------------------------------------------------------
# End-to-end graph_system create_agent run (mocked provider + file IO)
# ---------------------------------------------------------------------------


def _wire_stubs(tmp_path: Path, monkeypatch, agent_key: str = "test_agent") -> dict[str, str]:
    """Apply all stubs needed for a create_agent run. Returns a dict populated with written content."""
    monkeypatch.setattr(initialize_module, "get_agents", lambda: [])
    monkeypatch.setattr(initialize_module, "list_agent_keys", lambda: [])

    # select_tools_node: no registry tools in tests → it short-circuits to [].
    st_mod = importlib.import_module("solidcue.core.graph_system.nodes.select_tools_node")
    monkeypatch.setattr(st_mod, "_available_tools", lambda: [])

    async def _stream(messages, **_):
        target = "content"
        for m in messages:
            if m.get("role") == "user":
                txt = m["content"].lower()
                if "persona" in txt:
                    target = "Persona"
                elif "skill" in txt:
                    target = "Skill"
                elif "tools" in txt:
                    target = "Tools"
                break
        yield f"# Generated {target} Content"

    fake_provider = MagicMock()
    fake_provider.async_stream_generate = _stream
    monkeypatch.setattr(gen_mod, "_get_workspace_provider", lambda: fake_provider)

    monkeypatch.setattr(lc_mod, "SKILLS_ROOT_DIR", tmp_path)
    for tgt in ("persona", "skill", "tools"):
        (tmp_path / f"create-{tgt}.md").write_text(f"# Contract for {tgt}")

    written: dict[str, str] = {}

    def _make_saver(label):
        def _save(ak, content=None, *, overwrite=False):
            path = tmp_path / ak / f"{label}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content or "")
            written[label] = content or ""
            return path
        return _save

    # Patch at loader module level — write_node uses _loader.save_agent_*
    monkeypatch.setattr(loader_mod, "save_agent_persona", _make_saver("PERSONA"))
    monkeypatch.setattr(loader_mod, "save_agent_skill",   _make_saver("SKILL"))
    monkeypatch.setattr(loader_mod, "save_agent_tools",   _make_saver("TOOLS"))

    fake_config = AgentConfig(
        agent_key=agent_key,
        name="Test Agent",
        description="A test agent",
        provider=ProviderConfig(type="anthropic", api_key_env="K", model="m"),
    )
    monkeypatch.setattr(wc_mod, "write_agent_config", lambda _: (fake_config, str(tmp_path / f"{agent_key}.yaml")))

    monkeypatch.setattr(verify_mod, "load_agent", lambda key: fake_config)
    monkeypatch.setattr(verify_mod, "get_persona_path", lambda k: tmp_path / k / "PERSONA.md")
    monkeypatch.setattr(verify_mod, "get_skill_path",   lambda k: tmp_path / k / "SKILL.md")
    monkeypatch.setattr(verify_mod, "get_tools_path",   lambda k: tmp_path / k / "TOOLS.md")

    agent_dir = tmp_path / agent_key
    agent_dir.mkdir(exist_ok=True)
    for f in ("PERSONA.md", "SKILL.md", "TOOLS.md"):
        (agent_dir / f).write_text("placeholder")

    return written


def _build_async_graph():
    """Build system graph with InMemorySaver so ainvoke works in tests."""
    from langgraph.checkpoint.memory import InMemorySaver
    from solidcue.core.graph_system.builder import _compile_graph
    return _compile_graph(InMemorySaver())


@pytest.mark.asyncio
async def test_system_graph_create_agent_end_to_end(tmp_path, monkeypatch):
    written = _wire_stubs(tmp_path, monkeypatch)
    graph = _build_async_graph()
    result = await graph.ainvoke(
        {
            "thread_id": "e2e-1",
            "conversation_id": "e2e-conv-1",
            "user_input": "create a new agent",
            "agent_spec": _VALID_SPEC,
            "metadata": {},
        },
        config={"configurable": {"thread_id": "e2e-1"}},
    )

    assert result["system_intent"] == "create_agent"
    assert "PERSONA" in written
    assert "SKILL" in written
    assert "TOOLS" in written
    assert result.get("created_agent_key") == "test_agent"
    assert "test_agent" in result["final_response"]
    assert "successfully" in result["final_response"]


@pytest.mark.asyncio
async def test_system_graph_output_excludes_shared_message_channels(tmp_path, monkeypatch):
    _wire_stubs(tmp_path, monkeypatch)
    graph = _build_async_graph()
    result = await graph.ainvoke(
        {
            "thread_id": "e2e-1-output",
            "conversation_id": "e2e-conv-output",
            "user_input": "create a new agent",
            "agent_spec": _VALID_SPEC,
            "metadata": {},
        },
        config={"configurable": {"thread_id": "e2e-1-output"}},
    )

    assert "messages" not in result
    assert "chat_history" not in result


@pytest.mark.asyncio
async def test_system_graph_create_agent_artifacts_populated(tmp_path, monkeypatch):
    _wire_stubs(tmp_path, monkeypatch)
    graph = _build_async_graph()
    result = await graph.ainvoke(
        {
            "thread_id": "e2e-2",
            "user_input": "create a new agent",
            "agent_spec": _VALID_SPEC,
            "metadata": {},
        },
        config={"configurable": {"thread_id": "e2e-2"}},
    )

    artifacts = result.get("artifacts") or []
    targets = {a["target"] for a in artifacts}
    assert targets == {"persona", "skill", "tools"}


@pytest.mark.asyncio
async def test_system_graph_create_agent_definition_files_have_real_content(tmp_path, monkeypatch):
    """Generated content (not empty) must be written."""
    written = _wire_stubs(tmp_path, monkeypatch)
    graph = _build_async_graph()
    await graph.ainvoke(
        {
            "thread_id": "e2e-3",
            "user_input": "create a new agent",
            "agent_spec": _VALID_SPEC,
            "metadata": {},
        },
        config={"configurable": {"thread_id": "e2e-3"}},
    )

    for label in ("PERSONA", "SKILL", "TOOLS"):
        assert written.get(label, "").strip(), f"{label}.md was written empty"


# ---------------------------------------------------------------------------
# Domain-agnosticism check — swapping agent_key must not break anything
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_graph_domain_agnostic(tmp_path, monkeypatch):
    """The flow must work for any agent_key — no hardcoded domain strings in core."""
    spec = dict(_VALID_SPEC)
    spec["agent_key"] = "product_catalog"
    spec["name"] = "Product Catalog Agent"
    spec["description"] = "Manages product catalog data"

    written = _wire_stubs(tmp_path, monkeypatch, agent_key="product_catalog")
    graph = _build_async_graph()
    result = await graph.ainvoke(
        {
            "thread_id": "domain-check",
            "user_input": "create catalog agent",
            "agent_spec": spec,
            "metadata": {},
        },
        config={"configurable": {"thread_id": "domain-check"}},
    )

    assert result["created_agent_key"] == "product_catalog"
    assert "product_catalog" in result["final_response"]
    assert result["system_intent"] == "create_agent"
    assert "PERSONA" in written
    assert "SKILL" in written
    assert "TOOLS" in written


# ---------------------------------------------------------------------------
# Option A — interrupt-based spec collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_spec_interrupts_on_missing_fields(tmp_path, monkeypatch):
    """With no workspace provider to inherit, an incomplete spec pauses with a form."""
    _wire_stubs(tmp_path, monkeypatch)
    # No workspace provider → form fallback (don't depend on the real profile).
    cs_mod = importlib.import_module("solidcue.core.graph_system.nodes.collect_spec_node")
    monkeypatch.setattr(cs_mod, "_workspace_provider_defaults", lambda: None)

    graph = _build_async_graph()
    result = await graph.ainvoke(
        {
            "thread_id": "int-1",
            "user_input": "create a new agent",
            "agent_spec": {"name": "X"},
            "metadata": {},
        },
        config={"configurable": {"thread_id": "int-1"}},
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "collect_agent_spec"
    # Missing basic fields and provider fields are both surfaced.
    assert "agent_key" in payload["invalid_fields"]
    assert "description" in payload["invalid_fields"]
    assert "decision_provider_type" in payload["invalid_fields"]
    # The form schema tells the frontend how to render the secure form.
    schema = payload["form_schema"]
    assert "decision" in schema["provider_roles"]
    assert "api_key" in schema["secret_fields"]
    assert "anthropic" in schema["provider_types"]
    # Definition substance is elicited too: the artifacts question + key tasks.
    assert "produces_artifacts" in payload["gather_fields"]
    assert "key_tasks" in payload["gather_fields"]
    assert "produces_artifacts" in schema["definition"]
    # No agent was created while paused.
    assert not result.get("created_agent_key")


def test_collect_spec_inherits_workspace_provider(monkeypatch):
    """A name/key/description-only spec completes with no interrupt when a
    workspace provider is available to inherit."""
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

    result = cs_mod.collect_spec_node(
        {
            "agent_spec": {
                "name": "Weather",
                "agent_key": "weather_assistant",
                "description": "Checks weather",
                # Definition substance supplied so only provider inheritance is
                # under test here (not the new elicitation gate).
                "produces_artifacts": False,
                "key_tasks": ["report the forecast"],
            }
        }
    )

    assert result.get("system_next") != "final_output"
    assert result["created_agent_key"] == "weather_assistant"
    # Providers were inherited into the spec.
    assert result["agent_spec"]["decision_model"] == "m"


@pytest.mark.asyncio
async def test_collect_spec_resume_completes_creation(tmp_path, monkeypatch):
    """Resuming with a complete spec finishes the create_agent flow."""
    from langgraph.types import Command

    written = _wire_stubs(tmp_path, monkeypatch)
    graph = _build_async_graph()
    cfg = {"configurable": {"thread_id": "int-2"}}

    first = await graph.ainvoke(
        {
            "thread_id": "int-2",
            "user_input": "create a new agent",
            "agent_spec": {"name": "X"},
            "metadata": {},
        },
        config=cfg,
    )
    assert "__interrupt__" in first

    result = await graph.ainvoke(
        Command(resume={"agent_spec": _VALID_SPEC}),
        config=cfg,
    )

    assert result.get("created_agent_key") == "test_agent"
    assert "successfully" in result["final_response"]
    assert "PERSONA" in written
    assert "SKILL" in written
    assert "TOOLS" in written


@pytest.mark.asyncio
async def test_collect_spec_resume_still_incomplete_reasks(tmp_path, monkeypatch):
    """Resuming with a still-incomplete spec re-asks (interrupts again) instead of
    erroring — the gate loops until the spec is structurally valid."""
    from langgraph.types import Command

    _wire_stubs(tmp_path, monkeypatch)
    # No workspace provider → the reply must supply providers; keep it incomplete.
    cs_mod = importlib.import_module("solidcue.core.graph_system.nodes.collect_spec_node")
    monkeypatch.setattr(cs_mod, "_workspace_provider_defaults", lambda: None)
    graph = _build_async_graph()
    cfg = {"configurable": {"thread_id": "int-3"}}

    await graph.ainvoke(
        {
            "thread_id": "int-3",
            "user_input": "create a new agent",
            "agent_spec": {"name": "X"},
            "metadata": {},
        },
        config=cfg,
    )
    # Reply still omits required fields (description, providers) → gate re-asks.
    result = await graph.ainvoke(
        Command(resume={"agent_spec": {"name": "X", "agent_key": "x_agent"}}),
        config=cfg,
    )

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["type"] == "collect_agent_spec"
    assert "description" in result["__interrupt__"][0].value["invalid_fields"]
    assert not result.get("created_agent_key")
