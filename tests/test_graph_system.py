import importlib
from types import SimpleNamespace

from solidcue.core.graph_system.builder import build_system_graph
from solidcue.core.graph_system.nodes.initialize_node import initialize_node

initialize_module = importlib.import_module("solidcue.core.graph_system.nodes.initialize_node")


def test_system_initialize_reports_empty_workspace(monkeypatch) -> None:
    monkeypatch.setattr(initialize_module, "get_agents", lambda: [])
    monkeypatch.setattr(initialize_module, "list_agent_keys", lambda: [])

    result = initialize_node({"metadata": {}, "system_intent": "bootstrap"})

    assert result["workspace_has_agents"] is False
    assert result["available_agent_keys"] == []
    assert result["available_agents"] == []
    assert "create-agent" in result["available_system_skill_keys"]
    assert result["system_intent"] == "bootstrap"


def test_system_graph_guides_agent_creation_when_workspace_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(initialize_module, "get_agents", lambda: [])
    monkeypatch.setattr(initialize_module, "list_agent_keys", lambda: [])

    graph = build_system_graph()
    result = graph.invoke(
        {
            "thread_id": "thread-1",
            "conversation_id": "conversation-1",
            "user_input": "I want to set up the workspace",
            "metadata": {},
        },
        config={"configurable": {"thread_id": "thread-1"}},
    )

    assert result["system_intent"] == "create_agent"
    assert "No agents are configured yet" in result["final_response"]
    assert result["system_skill_key"] == "create-agent"
    assert result["system_skill_path"].endswith("create-agent.md")
    assert "Create Agent Skill" in result["system_skill"]


def test_system_graph_guides_selection_when_agents_exist(monkeypatch) -> None:
    monkeypatch.setattr(
        initialize_module,
        "get_agents",
        lambda: [SimpleNamespace(agent_key="resume_builder", name="Resume Builder", description="Build resumes")],
    )
    monkeypatch.setattr(initialize_module, "list_agent_keys", lambda: ["resume_builder"])

    graph = build_system_graph()
    result = graph.invoke(
        {
            "thread_id": "thread-2",
            "conversation_id": "conversation-2",
            "user_input": "what can I do next?",
            "metadata": {},
        },
        config={"configurable": {"thread_id": "thread-2"}},
    )

    assert result["system_intent"] == "select_agent"
    assert "Pick one to continue" in result["final_response"]
    assert result["system_skill_key"] == "user-profile"
    assert result["system_skill_path"].endswith("user-profile.md")
    assert "User Profile Skill" in result["system_skill"]
