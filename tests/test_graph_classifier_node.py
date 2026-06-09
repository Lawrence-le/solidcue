from solidcue.core.graph_node import classifier_node as classifier_node_module
from solidcue.core.graph_node.classifier_node import classifier_node


class _FakeAgent:
    name = "Resume Writer"
    description = "Writes resumes"


class _FakeProvider:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate(self, _messages, **_kwargs) -> str:
        return self.output


def test_classifier_node_resets_phase_to_source_for_task_intent(monkeypatch) -> None:
    monkeypatch.setattr(classifier_node_module, "load_agent", lambda _key: _FakeAgent())
    monkeypatch.setattr(classifier_node_module, "load_agent_persona", lambda _key: "persona")
    monkeypatch.setattr(
        classifier_node_module,
        "get_provider_for_role",
        lambda _agent, _role: _FakeProvider('{"intent":"task"}'),
    )

    result = classifier_node(
        {
            "agent_key": "resume_writer",
            "user_input": "start the task i asked you to work on",
            "phase": "conversational",
        }
    )

    assert result["phase"] == "source"
    assert result["messages"][0]["content"] == "Task intent detected — proceeding to discovery"
