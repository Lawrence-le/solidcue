from __future__ import annotations

from typing import Any, cast

from solidcue.core.graph_node.execution_node import _execute_tool
from solidcue.core.state.schema import AgentState


def artifact_execution_node(state: AgentState) -> dict[str, Any]:
    artifact_input = state.get("artifact_input")
    decision = cast(dict[str, Any], state.get("decision") or {})
    artifact_plan = cast(dict[str, Any], state.get("artifact_plan") or {})

    tool_name = decision.get("tool_name") or artifact_plan.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return {
            "artifact_result": {
                "success": False,
                "type": "artifact_execution",
                "content": None,
                "error": "Artifact tool is missing.",
            }
        }

    if not isinstance(artifact_input, dict) or not artifact_input:
        return {
            "artifact_result": {
                "success": False,
                "type": "artifact_execution",
                "content": None,
                "error": "Artifact input is missing.",
            }
        }

    artifact_decision = {
        "action": "use_tool",
        "tool_stage": "artifact",
        "tool_name": tool_name,
        "tool_input": artifact_input,
        "final_answer": None,
        "approval_preview": decision.get("approval_preview"),
    }
    execution_update = _execute_tool({**state, "decision": artifact_decision})
    execution_result = execution_update.get("execution_result")
    if not isinstance(execution_result, dict):
        execution_result = {
            "success": False,
            "type": "artifact_execution",
            "content": None,
            "error": "Artifact execution did not return a result.",
        }

    update = {
        "artifact_result": execution_result,
        "execution_result": execution_result,
    }
    if "messages" in execution_update:
        update["messages"] = execution_update["messages"]
    return update
