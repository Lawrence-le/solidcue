import importlib as _il; synthesis_module = _il.import_module("solidcue.core.graph_agent.nodes.synthesis_node")
from solidcue.core.graph_agent.nodes.synthesis_node import synthesis_node


_DISALLOWED_SYNTHESIS_WRITES = (
    "draft_output",
    "finalization_reason",
    "retry_reason",
    "latest_output",
    "failure_type",
    "final_response",
    "decision",
    "phase",
)


class _Agent:
    pass


class _Provider:
    def generate(self, messages, **kwargs):
        return "Polished response from synthesis."


class _CaptureProvider:
    def __init__(self):
        self.messages = None

    def generate(self, messages, **kwargs):
        self.messages = messages
        return "Corrected response from synthesis."


def _patch(monkeypatch, provider=None) -> None:
    monkeypatch.setattr(synthesis_module, "load_agent", lambda _: _Agent())
    monkeypatch.setattr(synthesis_module, "get_provider_for_role", lambda agent, role: provider or _Provider())
    monkeypatch.setattr(synthesis_module, "load_agent_persona", lambda _: "")


def test_synthesis_writes_only_synthesis_draft_from_artifact(monkeypatch) -> None:
    _patch(monkeypatch)

    result = synthesis_node(
        {
            "agent_key": "resume_builder",
        }
    )

    assert result["synthesis_draft"] == "Polished response from synthesis."
    assert "metric_synthesis" in result
    assert isinstance(result["metric_synthesis"], dict)
    for key in _DISALLOWED_SYNTHESIS_WRITES:
        assert key not in result, f"synthesis_node must not write '{key}'"


def test_synthesis_writes_only_synthesis_draft_from_decision_respond(monkeypatch) -> None:
    _patch(monkeypatch)

    result = synthesis_node(
        {
            "agent_key": "x",
            "decision": {"action": "respond", "final_answer": "Direct answer"},
        }
    )

    assert result["synthesis_draft"] == "Polished response from synthesis."
    for key in _DISALLOWED_SYNTHESIS_WRITES:
        assert key not in result


def test_synthesis_writes_only_synthesis_draft_from_execution(monkeypatch) -> None:
    _patch(monkeypatch)

    result = synthesis_node(
        {
            "agent_key": "x",
            "execution_result": {"success": True, "content": "Tool output"},
        }
    )

    assert result["synthesis_draft"] == "Polished response from synthesis."
    for key in _DISALLOWED_SYNTHESIS_WRITES:
        assert key not in result


def test_synthesis_falls_back_to_raw_material_when_llm_fails(monkeypatch) -> None:
    class _BadProvider:
        def generate(self, messages, **kwargs):
            raise RuntimeError("Provider error")

    _patch(monkeypatch, provider=_BadProvider())

    result = synthesis_node(
        {
            "agent_key": "x",
            "execution_result": {"success": True, "content": "Raw tool output"},
        }
    )

    assert "Raw tool output" in result["synthesis_draft"]


def test_synthesis_prompt_includes_actionable_validation_retry_reason(monkeypatch) -> None:
    provider = _CaptureProvider()
    _patch(monkeypatch, provider=provider)

    result = synthesis_node(
        {
            "agent_key": "x",
            "retry_reason": "Ungrounded claims: Redis and SQLite.",
            "execution_result": {"success": True, "content": "Python evidence"},
        }
    )

    assert result["synthesis_draft"] == "Corrected response from synthesis."
    user_prompt = provider.messages[2]["content"]
    assert "PREVIOUS_VALIDATION_FAILURE" in user_prompt
    assert "Ungrounded claims: Redis and SQLite." in user_prompt
    assert "Remove or revise any factual claim" in user_prompt
    assert "Do not add new factual claims" in user_prompt
