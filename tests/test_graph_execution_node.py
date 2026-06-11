import base64
import json
from typing import Any

from pytest import MonkeyPatch

import importlib as _il; execution_node_module = _il.import_module("solidcue.core.graph_agent.nodes.execution_node")
from solidcue.core.graph_agent.nodes.execution_node import _decode_file_content, execution_node
from solidcue.core.graph_agent.state.schema import AgentState
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


def test_execution_node_does_not_write_legacy_evidence(
    monkeypatch: MonkeyPatch,
) -> None:
    """Legacy evidence fields are not written by execution_node."""
    _patch_happy_path(monkeypatch)

    state: AgentState = {
        "agent_key": "assistant",
        "phase": "source",
        "decision": {
            "action": "use_tool",
            "thought": "Need search context.",
            "tool_name": "search_web",
            "tool_input": {"query": "Arsenal fixtures"},
        },
    }

    result = execution_node(state)
    legacy_key = "context" + "_evidence"
    assert legacy_key not in result


def test_decode_file_content_normalizes_google_docs_text_export() -> None:
    raw = "\ufeffCompany: General Assembly\r\n\r\nRole: Engineer\r\n".encode()
    payload = {
        "encoding": "base64",
        "mimeType": "text/plain",
        "content": base64.b64encode(raw).decode(),
    }

    result = _decode_file_content(json.dumps(payload))

    assert result == {
        "mimeType": "text/plain",
        "content": "Company: General Assembly\n\nRole: Engineer",
    }


def test_decode_file_content_keeps_binary_base64_payloads() -> None:
    payload = {
        "encoding": "base64",
        "mimeType": "application/pdf",
        "content": base64.b64encode(b"%PDF\r\n\xff").decode(),
    }

    result = _decode_file_content(payload)

    assert result == payload


def test_execution_tool_history_records_task_id(monkeypatch: MonkeyPatch) -> None:
    _patch_happy_path(monkeypatch)
    state: AgentState = {
        "agent_key": "assistant",
        "current_task": "task_3",
        "phase": "source",
        "active_tool_call": {
            "action": "use_tool",
            "thought": "Need search context.",
            "tool_name": "search_web",
            "tool_input": {"query": "Arsenal fixtures"},
        },
    }

    result = execution_node(state)
    history = result.get("tool_call_history")
    assert isinstance(history, list) and history
    assert history[-1].get("task_id") == "task_3"


def test_execution_tool_history_uses_current_execution_output(monkeypatch: MonkeyPatch) -> None:
    _patch_happy_path(monkeypatch)

    fresh_base64 = base64.b64encode(b"fresh-docx-bytes").decode()

    class _ArtifactClient:
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
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "success",
                                "filename": "resume.docx",
                                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                "content_base64": fresh_base64,
                            }
                        ),
                    }
                ],
                "structured_content": None,
                "is_error": False,
            }

    monkeypatch.setattr(execution_node_module, "MCPClient", _ArtifactClient)
    class _ArtifactAgent:
        tools: list[str] = ["create_formatted_word_document_base64"]

    monkeypatch.setattr(execution_node_module, "load_agent", lambda _: _ArtifactAgent())
    monkeypatch.setattr(
        execution_node_module,
        "load_tool",
        lambda _: ToolConfig(
            tool_key="create_formatted_word_document_base64",
            name="Create formatted doc",
            type="mcp",
            mcp=MCPToolConfig(server_key="search_server", tool_name="create_formatted_word_document_base64"),
        ),
    )

    state: AgentState = {
        "agent_key": "assistant",
        "current_task": "task_5",
        "phase": "artifact",
        "active_tool_call": {
            "action": "use_tool",
            "thought": "Create formatted DOCX.",
            "tool_name": "create_formatted_word_document_base64",
            "tool_input": {"title": "Resume", "document_body_markdown": "content"},
        },
        # stale previous turn result should not be used
        "execution_result": {
            "success": True,
            "type": "tool_execution",
            "content": {"status": "success", "filename": "old.docx"},
            "error": None,
        },
    }

    result = execution_node(state)
    history = result.get("tool_call_history")
    assert isinstance(history, list) and history
    latest_execution_result = history[-1].get("execution_result")
    assert isinstance(latest_execution_result, dict)
    assert latest_execution_result.get("success") is True
    content = latest_execution_result.get("content")
    assert isinstance(content, dict)
    assert content.get("content_base64") == fresh_base64


def test_execution_fills_large_payload_from_required_handoff_key(monkeypatch: MonkeyPatch) -> None:
    class _UploadAgent:
        tools: list[str] = ["drive_upload_file"]

    captured_arguments: dict[str, Any] = {}

    class _UploadClient:
        def __init__(self, server: MCPServerConfig) -> None:
            self.server = server

        async def call_tool(
            self,
            *,
            tool_name: str,
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            captured_arguments.update(arguments)
            return {
                "tool_name": tool_name,
                "content": [{"type": "text", "text": json.dumps({"status": "ok"})}],
                "structured_content": None,
                "is_error": False,
            }

    expected_base64 = base64.b64encode(b"docx-bytes").decode()
    wrong_base64 = base64.b64encode(b"wrong-bytes").decode()

    monkeypatch.setattr(execution_node_module, "load_agent", lambda _: _UploadAgent())
    monkeypatch.setattr(
        execution_node_module,
        "load_tool",
        lambda _: ToolConfig(
            tool_key="drive_upload_file",
            name="Drive Upload",
            type="mcp",
            mcp=MCPToolConfig(
                server_key="search_server",
                tool_name="drive_upload_file",
                input_schema={"type": "object", "properties": {"content_base64": {"type": "string"}}},
            ),
        ),
    )
    monkeypatch.setattr(execution_node_module, "MCPClient", _UploadClient)
    monkeypatch.setattr(
        execution_node_module,
        "load_mcp_server",
        lambda _: MCPServerConfig(
            server_key="search_server",
            name="Search Server",
            url="http://localhost:9000/mcp",
        ),
    )

    state: AgentState = {
        "agent_key": "assistant",
        "phase": "artifact",
        "current_task": "task_8",
        "task_plan": [
            {"id": "task_8", "requires": ["resume_uploaded_to_drive"], "context": {"item_key": "u_1"}},
        ],
        "handoff": {
            "word_document_payload_generated::u_1": {"content_base64": expected_base64},
            # Current task output key should never be used as input source.
            "resume_uploaded_to_drive::u_1": {"content_base64": wrong_base64},
        },
        "active_tool_call": {
            "action": "use_tool",
            "tool_name": "drive_upload_file",
            "tool_input": {"content_base64": "[payload via handoff]"},
        },
    }

    result = execution_node(state)
    assert isinstance(result.get("execution_result"), dict)
    assert captured_arguments.get("content_base64") == expected_base64
