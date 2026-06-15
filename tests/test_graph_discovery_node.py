import pytest

from solidcue.core.graph_agent.nodes.discovery_node import discovery_node
import importlib as _il; discovery_module = _il.import_module("solidcue.core.graph_agent.nodes.discovery_node")


@pytest.mark.asyncio
async def test_discovery_node_extracts_paths_from_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        discovery_module,
        "load_agent_skill",
        lambda _agent_key: "Load from resume_agent/source/experience.md and resume_agent/source/project_bank.md",
    )
    monkeypatch.setattr(
        discovery_module,
        "load_agent_persona",
        lambda _agent_key: "persona content",
    )
    monkeypatch.setattr(discovery_module, "load_agent_tools", lambda _agent_key: "tools content")

    class _Agent:
        pass

    class _Provider:
        async def async_stream_generate(self, _messages, **kwargs):
            yield (
                '{"source_paths":["resume_agent/source/experience.md","resume_agent/source/project_bank.md"],'
                '"output_paths":["resume_agent/generated_resumes/"],'
                '"source_filenames":["experience.md","project_bank.md"],'
                '"output_filenames":["Lawrence Lee Resume.docx"]}'
            )

    monkeypatch.setattr(discovery_module, "load_agent", lambda _agent_key: _Agent())
    monkeypatch.setattr(discovery_module, "get_provider_for_role", lambda _agent, _role: _Provider())

    result = await discovery_node({"agent_key": "resume_builder"})

    assert result["source_paths"] == [
        "resume_agent/source/experience.md",
        "resume_agent/source/project_bank.md",
    ]
    assert result["metadata"]["source_paths"] == result["source_paths"]
    assert result["source_filenames"] == ["experience.md", "project_bank.md"]
    assert result["output_filenames"] == ["Lawrence Lee Resume.docx"]
    assert result["metadata"]["source_filenames"] == result["source_filenames"]
    assert result["metadata"]["output_filenames"] == result["output_filenames"]
    assert "metric_discovery" in result
    assert isinstance(result["metric_discovery"], dict)


@pytest.mark.asyncio
async def test_discovery_node_uses_legacy_llm_paths_key_as_source_paths(monkeypatch) -> None:
    monkeypatch.setattr(discovery_module, "load_agent_skill", lambda _agent_key: "")
    monkeypatch.setattr(
        discovery_module,
        "load_agent_persona",
        lambda _agent_key: "Use resume source files and project files too.",
    )
    monkeypatch.setattr(discovery_module, "load_agent_tools", lambda _agent_key: "")

    class _Agent:
        pass

    class _Provider:
        async def async_stream_generate(self, _messages, **kwargs):
            yield '{"paths":["resume_agent/source/experience.md","resume_agent/source/project_bank.md"]}'

    monkeypatch.setattr(discovery_module, "load_agent", lambda _agent_key: _Agent())
    monkeypatch.setattr(discovery_module, "get_provider_for_role", lambda _agent, _role: _Provider())

    result = await discovery_node({"agent_key": "resume_builder"})

    assert result["source_paths"] == [
        "resume_agent/source/experience.md",
        "resume_agent/source/project_bank.md",
    ]
    assert result["source_filenames"] == []
    assert result["output_filenames"] == []


@pytest.mark.asyncio
async def test_discovery_node_prefers_skill_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        discovery_module,
        "load_agent_skill",
        lambda _agent_key: (
            "- Source path `resume_agent/source/profile`\n"
            "- Source path `resume_agent/source/experience`\n"
        ),
    )
    monkeypatch.setattr(discovery_module, "load_agent_persona", lambda _agent_key: "")
    monkeypatch.setattr(discovery_module, "load_agent_tools", lambda _agent_key: "")

    class _Agent:
        pass

    class _Provider:
        async def async_stream_generate(self, _messages, **kwargs):
            yield '{"source_paths":["resume_agent/source/profile","resume_agent/source/experience"],"output_paths":[]}'

    monkeypatch.setattr(discovery_module, "load_agent", lambda _agent_key: _Agent())
    monkeypatch.setattr(discovery_module, "get_provider_for_role", lambda _agent, _role: _Provider())

    result = await discovery_node({"agent_key": "resume_builder"})

    assert result["source_paths"] == ["resume_agent/source/profile", "resume_agent/source/experience"]
    assert result["source_filenames"] == []
    assert result["output_filenames"] == []


@pytest.mark.asyncio
async def test_discovery_node_extracts_paths_from_tools_md(monkeypatch) -> None:
    monkeypatch.setattr(discovery_module, "load_agent_skill", lambda _agent_key: "")
    monkeypatch.setattr(discovery_module, "load_agent_persona", lambda _agent_key: "")
    monkeypatch.setattr(
        discovery_module,
        "load_agent_tools",
        lambda _agent_key: (
            "Use `drive_list_by_path` first to locate source files under:\n"
            "- `resume_agent/source/master`\n"
            "Then upload to `resume_agent/generated_resumes/`\n"
        ),
    )

    class _Agent:
        pass

    class _Provider:
        async def async_stream_generate(self, _messages, **kwargs):
            yield (
                '{"source_paths":["resume_agent/source/master"],'
                '"output_paths":["resume_agent/generated_resumes/"]}'
            )

    monkeypatch.setattr(discovery_module, "load_agent", lambda _agent_key: _Agent())
    monkeypatch.setattr(discovery_module, "get_provider_for_role", lambda _agent, _role: _Provider())

    result = await discovery_node({"agent_key": "resume_builder"})

    assert result["source_paths"] == ["resume_agent/source/master"]
    assert result["output_paths"] == ["resume_agent/generated_resumes/"]
    assert result["source_filenames"] == []
    assert result["output_filenames"] == []
