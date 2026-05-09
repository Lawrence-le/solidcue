from solidcue.core.graph_node.discovery_node import discovery_node
from solidcue.core.graph_node import discovery_node as discovery_module


def test_discovery_node_extracts_persona_drive_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        discovery_module,
        "load_agent_persona",
        lambda _agent_key: (
            '[retrieve from: Google Drive path "resume_agent/source/experience.md"]\n'
            '[retrieve from: Google Drive path "resume_agent/source/project_bank.md"]'
        ),
    )

    result = discovery_node({"agent_key": "resume_builder"})

    assert result["persona_source_paths"] == [
        "resume_agent/source/experience.md",
        "resume_agent/source/project_bank.md",
    ]
    assert result["metadata"]["persona_source_paths"] == result["persona_source_paths"]


def test_discovery_node_uses_llm_fallback_when_deterministic_has_no_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        discovery_module,
        "load_agent_persona",
        lambda _agent_key: "Use resume source files under resume_agent/source and project files too.",
    )

    class _Agent:
        pass

    class _Provider:
        def generate(self, _messages):
            return '{"paths":["resume_agent/source/experience.md","resume_agent/source/project_bank.md"]}'

    monkeypatch.setattr(discovery_module, "load_agent", lambda _agent_key: _Agent())
    monkeypatch.setattr(discovery_module, "get_provider_for_role", lambda _agent, _role: _Provider())

    result = discovery_node({"agent_key": "resume_builder"})

    assert result["persona_source_paths"] == [
        "resume_agent/source/experience.md",
        "resume_agent/source/project_bank.md",
    ]
