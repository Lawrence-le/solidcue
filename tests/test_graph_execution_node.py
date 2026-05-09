from typing import Any

from pytest import MonkeyPatch

from solidcue.core.graph_node import execution_node as execution_node_module
from solidcue.core.graph_node.execution_node import execution_node
from solidcue.core.state.schema import AgentState
from solidcue.tools.schema import MCPServerConfig, MCPToolConfig, ToolConfig


class _Agent:
    tools: list[str] = ["search_web"]


class _MCPClient:
    def __init__(self, server: MCPServerConfig) -> None:
        self.server = server

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "content": [{"text": f"result for {arguments['query']}"}],
            "structured_content": None,
            "is_error": False,
        }


def _patch_happy_path(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(execution_node_module, "load_agent", lambda _: _Agent())
    monkeypatch.setattr(
        execution_node_module,
        "load_tool",
        lambda _: ToolConfig(
            tool_key="search_web",
            name="Search Web",
            type="mcp",
            mcp=MCPToolConfig(server_key="search_server", tool_name="search"),
        ),
    )
    monkeypatch.setattr(
        execution_node_module,
        "load_mcp_server",
        lambda _: MCPServerConfig(
            server_key="search_server",
            name="Search Server",
            url="http://localhost:9000/mcp",
        ),
    )
    monkeypatch.setattr(execution_node_module, "MCPClient", _MCPClient)


def test_execution_node_appends_context_evidence_for_successful_context_tool(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_happy_path(monkeypatch)

    state: AgentState = {
        "agent_key": "assistant",
        "attempt": 0,
        "decision": {
            "action": "use_tool",
            "tool_stage": "context",
            "thought": "Need search context.",
            "tool_name": "search_web",
            "tool_input": {"query": "Arsenal fixtures"},
            "final_answer": None,
        },
    }

    result = execution_node(state)
    evidence = result.get("context_evidence")

    assert isinstance(evidence, list)
    assert len(evidence) == 1
    assert evidence[0]["tool_name"] == "search_web"
    assert evidence[0]["tool_input"] == {"query": "Arsenal fixtures"}
    assert evidence[0]["content"] == "result for Arsenal fixtures"


def test_execution_node_does_not_append_context_evidence_for_artifact_tool(
    monkeypatch: MonkeyPatch,
) -> None:
    _patch_happy_path(monkeypatch)

    state: AgentState = {
        "agent_key": "assistant",
        "attempt": 0,
        "decision": {
            "action": "use_tool",
            "tool_stage": "artifact",
            "thought": "Generate artifact.",
            "tool_name": "search_web",
            "tool_input": {"query": "Arsenal fixtures"},
            "final_answer": None,
        },
    }

    result = execution_node(state)
    assert "context_evidence" not in result
