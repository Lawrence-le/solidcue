from types import SimpleNamespace

from solidcue.services import agent_service as agent_service_module
from solidcue.services.agent_service import CreateAgentInput


class _FakeGraph:
    def __init__(self) -> None:
        self.invoked_with = None
        self.invoked_config = None

    def invoke(self, state, config=None):
        self.invoked_with = state
        self.invoked_config = config
        return {**state, "final_output": "ok", "workflow_status": "completed"}


def test_run_agent_uses_langgraph_invoke(monkeypatch) -> None:
    fake_agent = SimpleNamespace(agent_key="generic_assistant")
    fake_profile = SimpleNamespace(model_dump=lambda exclude_none=True: {"location": "Singapore"})
    fake_graph = _FakeGraph()

    monkeypatch.setattr(agent_service_module, "load_agent", lambda _: fake_agent)
    monkeypatch.setattr(agent_service_module, "load_user_profile", lambda: fake_profile)
    monkeypatch.setattr(agent_service_module, "build_agent_graph", lambda: fake_graph)

    agent, result = agent_service_module.run_agent(
        "generic_assistant",
        "hello",
        thread_id="thread-123",
        debug=False,
    )

    assert agent.agent_key == "generic_assistant"
    assert fake_graph.invoked_with is not None
    assert fake_graph.invoked_with["agent_key"] == "generic_assistant"
    assert fake_graph.invoked_with["user_input"] == "hello"
    assert fake_graph.invoked_with["config"] == {"location": "Singapore"}
    assert fake_graph.invoked_with["max_retries"] == 10
    assert fake_graph.invoked_config["run_name"] == "solidcue:generic_assistant"
    assert "solidcue" in fake_graph.invoked_config["tags"]
    assert fake_graph.invoked_config["metadata"]["agent_key"] == "generic_assistant"
    assert fake_graph.invoked_config["metadata"]["location"] == "Singapore"
    assert fake_graph.invoked_config["configurable"]["thread_id"] == "thread-123"
    assert result["workflow_status"] == "completed"


def test_create_agent_creates_persona_file(monkeypatch) -> None:
    captured = {"saved_agent_key": None}

    monkeypatch.setattr(agent_service_module, "write_env_key", lambda *_: None)
    monkeypatch.setattr(agent_service_module, "generate_env_key", lambda key: f"{key.upper()}_API_KEY")

    def _save_agent(config):
        captured["saved_agent_key"] = config.agent_key
        return "/tmp/generic_assistant.yaml"

    def _save_agent_persona(agent_key: str):
        captured["persona_key"] = agent_key
        return "/tmp/generic_assistant/PERSONA.md"

    monkeypatch.setattr(agent_service_module, "save_agent", _save_agent)
    monkeypatch.setattr(agent_service_module, "save_agent_persona", _save_agent_persona)

    input_data = CreateAgentInput(
        name="Generic Assistant",
        agent_key="generic_assistant",
        description="Agent",
        decision_provider_type="openai_compatible",
        decision_base_url="http://localhost:11434/v1",
        decision_api_key="x",
        decision_model="model-a",
        sufficiency_provider_type="openai_compatible",
        sufficiency_base_url="http://localhost:11434/v1",
        sufficiency_api_key="y",
        sufficiency_model="model-b",
        validator_provider_type="openai_compatible",
        validator_base_url="http://localhost:11434/v1",
        validator_api_key="z",
        validator_model="model-c",
        selected_tools=[],
    )

    config, _ = agent_service_module.create_agent(input_data)
    assert config.agent_key == "generic_assistant"
    assert captured["saved_agent_key"] == "generic_assistant"
    assert captured["persona_key"] == "generic_assistant"
