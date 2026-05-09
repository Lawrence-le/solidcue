import asyncio
import json
from typing import Any, cast

from solidcue.agents.configs.loader import load_agent
from solidcue.app.utils.helpers import normalize_tool_output
from solidcue.core.state.schema import AgentState
from solidcue.tools.loader import load_mcp_server, load_tool
from solidcue.tools.mcp.client import MCPClient


def _execution_result(success: bool, result_type: str, content: Any, error: Any) -> dict[str, Any]:
    return {
        "success": success,
        "type": result_type,
        "content": content,
        "error": error,
    }


def _record_tool_call(state: AgentState) -> list[dict[str, Any]]:
    decision = state.get("active_tool_call") or state.get("decision")
    if not isinstance(decision, dict):
        return []

    tool_name = decision.get("tool_name")
    tool_input = decision.get("tool_input")
    if not isinstance(tool_name, str) or not tool_name:
        return []

    history = state.get("tool_call_history")
    normalized_history = history if isinstance(history, list) else []
    normalized_tool_input = tool_input if isinstance(tool_input, dict) else {}
    return [
        *normalized_history,
        {
            "tool_name": tool_name,
            "tool_input": normalized_tool_input,
            "signature": (
                f"{tool_name}:"
                f"{json.dumps(normalized_tool_input, sort_keys=True, ensure_ascii=True, default=str)}"
            ),
        },
    ]


def _build_source_evidence_entry(
    state: AgentState,
    execution_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a single new evidence entry, or None if execution didn't produce material content.

    Caller is responsible for appending into context_evidence (legacy, full overwrite)
    and source_evidence (redesign, append-only via operator.add reducer).
    """
    decision = state.get("active_tool_call") or state.get("decision")
    if not isinstance(decision, dict) or decision.get("tool_stage") != "context":
        return None

    if not isinstance(execution_result, dict) or execution_result.get("success") is not True:
        return None

    content = execution_result.get("content")
    if content is None:
        return None
    content_text = str(content).strip()
    if not content_text:
        return None

    tool_name = decision.get("tool_name")
    tool_input = decision.get("tool_input")
    return {
        "tool_name": tool_name if isinstance(tool_name, str) else "",
        "tool_input": tool_input if isinstance(tool_input, dict) else {},
        "content": content_text,
    }


def _append_context_evidence(
    state: AgentState,
    new_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Legacy: build full context_evidence list by appending new entry to prior state."""
    existing = state.get("context_evidence")
    history = list(existing) if isinstance(existing, list) else []
    history.append(new_entry)
    return history


def _source_id_from_input(tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in ("file_id", "document_id", "spreadsheet_id", "id"):
        value = tool_input.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _extract_files(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return []
    if isinstance(content, dict):
        files = content.get("files")
        if isinstance(files, list):
            return [file for file in files if isinstance(file, dict)]
    return []


def _update_source_manifest(
    state: AgentState,
    execution_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(execution_result, dict) or execution_result.get("success") is not True:
        return None

    decision = state.get("active_tool_call") or state.get("decision")
    if not isinstance(decision, dict) or decision.get("tool_stage") != "context":
        return None

    manifest = dict(state.get("source_manifest") or {})
    sources = list(manifest.get("sources") or [])
    by_id = {
        str(source.get("id")): dict(source)
        for source in sources
        if isinstance(source, dict) and source.get("id") is not None
    }

    content = execution_result.get("content")
    for file in _extract_files(content):
        file_id = file.get("id") or file.get("file_id") or file.get("document_id")
        if file_id is None:
            continue
        file_id_text = str(file_id).strip()
        if not file_id_text:
            continue
        existing = by_id.get(file_id_text, {})
        existing.update(
            {
                "id": file_id_text,
                "name": str(file.get("name") or existing.get("name") or ""),
                "uri": str(file.get("webViewLink") or existing.get("uri") or ""),
                "mime_type": str(file.get("mimeType") or existing.get("mime_type") or ""),
                "status": existing.get("status") or "listed",
                "read_attempts": int(existing.get("read_attempts", 0)),
            }
        )
        by_id[file_id_text] = existing

    read_id = _source_id_from_input(decision.get("tool_input"))
    if read_id and content is not None and str(content).strip():
        existing = by_id.get(read_id, {"id": read_id, "read_attempts": 0})
        existing["status"] = "read"
        existing["read_attempts"] = int(existing.get("read_attempts", 0)) + 1
        by_id[read_id] = existing

    if not by_id:
        return None

    return {"sources": list(by_id.values())}

def _execute_tool(state: AgentState) -> dict[str, Any]:
    decision = cast(dict[str, Any], state.get("active_tool_call") or state.get("decision") or {})

    action = decision.get("action")

    if action != "use_tool":
        return {
            "execution_result": _execution_result(
                success=True,
                result_type="skipped",
                content=None,
                error=None,
            )
        }

    tool_key = decision.get("tool_name")
    arguments = decision.get("tool_input") or {}
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {
            "execution_result": _execution_result(
                success=False,
                result_type="tool_execution",
                content=None,
                error="Execution Error: agent_key missing",
            )
        }

    try:
        agent = load_agent(agent_key)
        if tool_key not in set(agent.tools or []):
            raise ValueError(f"Tool '{tool_key}' not allowed for agent '{agent_key}'")

        selected_tool = load_tool(tool_key)
        if selected_tool.type != "mcp" or not selected_tool.mcp:
            raise ValueError(f"Unsupported or misconfigured tool type: {selected_tool.type}")

        server = load_mcp_server(selected_tool.mcp.server_key)
        client = MCPClient(server)

        raw_output = asyncio.run(
            client.call_tool(
                tool_name=selected_tool.mcp.tool_name,
                arguments=arguments,
            )
        )

        normalized_output = normalize_tool_output(raw_output)
        is_tool_error = bool(raw_output.get("is_error"))
        if is_tool_error and "Unable to reach Open-Meteo service" in normalized_output:
            normalized_output = (
                f"{normalized_output}. The MCP server is running, but its outbound network to Open-Meteo failed."
            )

        update: dict[str, Any] = {
            "execution_result": _execution_result(
                success=not is_tool_error,
                result_type="tool_execution",
                content=normalized_output,
                error=normalized_output if is_tool_error else None,
            )
        }

        existing_messages = state.get("messages", [])
        if isinstance(existing_messages, list) and existing_messages:
            last_msg = existing_messages[-1]
            tool_calls = last_msg.get("tool_calls", []) if isinstance(last_msg, dict) else []
            tool_call_id = tool_calls[0].get("id") if tool_calls else f"tool_call_{state.get('attempt', 0)}"

            # LangGraph compatibility: return only newly produced message delta.
            update["messages"] = [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(raw_output, default=str),
                }
            ]

        return update

    except Exception as exc:
        error_text = str(exc)
        if "Unable to reach MCP server" in error_text:
            error_text = f"{error_text}. Check that the MCP service is running and reachable."

        return {
            "execution_result": _execution_result(
                success=False,
                result_type="tool_execution",
                content=None,
                error=f"Execution Error: {error_text}",
            )
        }


def execution_node(state: AgentState) -> dict[str, Any]:
    update: dict[str, Any] = {
        "tool_call_history": _record_tool_call(state),
    }

    execution_update = _execute_tool(state)
    update.update(execution_update)

    tool_turn_count_value = state.get("tool_turn_count")
    tool_turn_count = tool_turn_count_value if isinstance(tool_turn_count_value, int) else 0
    update["tool_turn_count"] = tool_turn_count + 1

    attempt_value = state.get("attempt")
    attempt = attempt_value if isinstance(attempt_value, int) else 0
    update["attempt"] = attempt + 1

    execution_result = execution_update.get("execution_result")
    if isinstance(execution_result, dict):
        new_entry = _build_source_evidence_entry(state, execution_result)
        if new_entry is not None:
            # Legacy: full list overwrite for context_evidence (still read by artifact_generation_node)
            update["context_evidence"] = _append_context_evidence(state, new_entry)
            # Redesign: append-only via operator.add reducer — return delta only
            update["source_evidence"] = [new_entry]
        source_manifest = _update_source_manifest(state, execution_result)
        if source_manifest is not None:
            update["source_manifest"] = source_manifest
        if execution_result.get("success") is False:
            failure_reason = execution_result.get("error") or execution_result.get("content")
            update["retry_reason"] = str(failure_reason) if failure_reason else "Tool execution failed."
        else:
            update["retry_reason"] = None

    return update
