import importlib as _il; decision_node_module = _il.import_module("solidcue.core.graph_agent.nodes.decision_node")
from solidcue.core.graph_agent.nodes.decision_node import decision_node
import pytest


class _FakeProvider:
    def __init__(self, output: str) -> None:
        self._output = output

    async def async_stream_generate(self, messages, **kwargs):
        yield self._output


class _FakeAgentConfig:
    def __init__(self, tools: list[str]) -> None:
        self.tools = tools


def _patch(monkeypatch, *, tools: list[str], output: str) -> None:
    monkeypatch.setattr(decision_node_module, "load_agent", lambda _: _FakeAgentConfig(tools))
    monkeypatch.setattr(decision_node_module, "build_decision_messages", lambda **kwargs: [])
    monkeypatch.setattr(decision_node_module, "get_provider_for_role", lambda agent, role: _FakeProvider(output))


@pytest.mark.asyncio
async def test_graph_decision_node_malformed_output_falls_back_to_respond(monkeypatch) -> None:
    """Malformed LLM output that cannot be parsed falls back to respond action."""
    malformed_output = (
        '{"thought":"Need csv.","action":"use_tool","tool_name":"create_csv_file",'
        '"tool_input":{"content":"sku,name\\nSKU001,A","title":"Inventory",'
        '"approval_preview":null}'
    )
    _patch(monkeypatch, tools=["create_csv_file"], output=malformed_output)

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "create csv"})

    assert result["tool_use"] is False
    assert result["decision"]["action"] == "respond"
    assert "metric_decision" in result
    assert isinstance(result["metric_decision"], dict)


@pytest.mark.asyncio
async def test_graph_decision_node_falls_back_when_tool_not_allowed(monkeypatch) -> None:
    _patch(
        monkeypatch,
        tools=["search_web"],
        output='{"thought":"Need tool.","action":"use_tool","tool_name":"create_csv_file","tool_input":{"content":"x","title":"y"}}',
    )

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "create csv"})

    assert result["tool_use"] is False
    assert result["decision"]["action"] == "respond"
    assert "couldn't safely execute" in result["decision"]["thought"]


@pytest.mark.asyncio
async def test_graph_decision_node_prefixed_json_still_responds(monkeypatch) -> None:
    _patch(
        monkeypatch,
        tools=[],
        output='Response: {"thought":"No tool needed","action":"respond","tool_name":null,"tool_input":null,"approval_preview":null}',
    )

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "hello"})

    assert result["tool_use"] is False
    assert result["decision"]["action"] == "respond"
    assert result["decision"]["thought"] == "No tool needed"


@pytest.mark.asyncio
async def test_graph_decision_node_rejects_use_tool_when_required_input_missing(monkeypatch) -> None:
    _patch(
        monkeypatch,
        tools=["scrape_webpage"],
        output='{"thought":"Need tool.","action":"use_tool","tool_name":"scrape_webpage","tool_input":{},"approval_preview":null}',
    )

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "summarize this page for me"})

    assert result["tool_use"] is False
    assert result["decision"]["action"] == "respond"
    assert "required inputs were missing" in result["decision"]["thought"]


@pytest.mark.asyncio
async def test_graph_decision_node_writes_active_tool_call_when_phase_is_source(monkeypatch) -> None:
    """When phase=source, tool call goes to active_tool_call not artifact_plan."""
    _patch(
        monkeypatch,
        tools=["search_web"],
        output='{"thought":"Need search.","action":"use_tool","tool_name":"search_web","tool_input":{"query":"latest llm"}}',
    )

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "latest llm", "phase": "source"})

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_name"] == "search_web"
    assert "active_tool_call" in result
    assert "artifact_plan" not in result


@pytest.mark.asyncio
async def test_graph_decision_node_writes_active_tool_call_when_phase_is_artifact(monkeypatch) -> None:
    """When phase=artifact, tool call still goes to active_tool_call."""
    _patch(
        monkeypatch,
        tools=["create_word_document"],
        output='{"thought":"Need document.","action":"use_tool","tool_name":"create_word_document","tool_input":{}}',
    )

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "create a resume document", "phase": "artifact"})

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_name"] == "create_word_document"
    assert "active_tool_call" in result
    assert "artifact_plan" not in result


_DISALLOWED_DECISION_WRITES = (
    "phase",
    "draft_output",
    "finalization_reason",
    "retry_reason",
    "latest_output",
    "failure_type",
    "synthesis_draft",
    "final_response",
)


@pytest.mark.asyncio
async def test_graph_decision_node_writes_active_tool_call_for_artifact_intent(monkeypatch) -> None:
    _patch(
        monkeypatch,
        tools=["create_word_document"],
        output='{"thought":"Need document.","action":"use_tool","tool_name":"create_word_document","tool_input":{"title":"Resume"},"approval_preview":null}',
    )

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "create a resume document", "phase": "artifact"})

    assert result["active_tool_call"]["tool_name"] == "create_word_document"
    assert result["active_tool_call"]["tool_input"] == {"title": "Resume"}
    assert "thought" in result["active_tool_call"]
    for key in _DISALLOWED_DECISION_WRITES:
        assert key not in result, f"decision_node must not write '{key}'"


@pytest.mark.asyncio
async def test_graph_decision_node_omits_disallowed_keys_for_respond_action(monkeypatch) -> None:
    _patch(
        monkeypatch,
        tools=[],
        output='{"thought":"No tool needed","action":"respond","tool_name":null,"tool_input":null,"approval_preview":null}',
    )

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "hello"})

    assert result["tool_use"] is False
    for key in _DISALLOWED_DECISION_WRITES:
        assert key not in result, f"decision_node must not write '{key}'"


@pytest.mark.asyncio
async def test_graph_decision_node_omits_disallowed_keys_for_source_tool_intent(monkeypatch) -> None:
    _patch(
        monkeypatch,
        tools=["search_web"],
        output='{"thought":"Need search.","action":"use_tool","tool_name":"search_web","tool_input":{"query":"x"},"approval_preview":null}',
    )

    result = await decision_node({"agent_key": "generic_assistant", "user_input": "x"})

    assert result["tool_use"] is True
    assert result["active_tool_call"]["tool_name"] == "search_web"
    for key in _DISALLOWED_DECISION_WRITES:
        assert key not in result, f"decision_node must not write '{key}'"


@pytest.mark.asyncio
async def test_graph_decision_node_filters_tool_call_history_to_current_task(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(decision_node_module, "load_agent", lambda _: _FakeAgentConfig(["search_web"]))
    monkeypatch.setattr(
        decision_node_module,
        "build_decision_messages",
        lambda **kwargs: captured.setdefault("history", kwargs.get("tool_call_history")) or [],
    )
    monkeypatch.setattr(
        decision_node_module,
        "get_provider_for_role",
        lambda agent, role: _FakeProvider('{"action":"respond","thought":"ok","tool_name":null,"tool_input":null}'),
    )

    await decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "x",
            "current_task": "task_2",
            "tool_call_history": [
                {"task_id": "task_1", "tool_name": "search_web", "tool_input": {"query": "a"}, "success": True},
                {"task_id": "task_2", "tool_name": "search_web", "tool_input": {"query": "b"}, "success": False},
                {"task_id": None, "tool_name": "search_web", "tool_input": {"query": "c"}, "success": False},
            ],
        }
    )

    history = captured.get("history")
    assert isinstance(history, list)
    # Includes current task attempts + successful prior entries
    assert len(history) == 2
    task_ids = [entry.get("task_id") for entry in history]
    assert "task_2" in task_ids
    assert "task_1" in task_ids
