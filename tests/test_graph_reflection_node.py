from solidcue.core.graph_node.reflection_node import reflection_node


def test_reflection_returns_missing_source_on_empty_source_content() -> None:
    result = reflection_node(
        {
            "phase": "source",
            "current_task": "task_1",
            "decision": {"action": "use_tool", "tool_name": "browser_get_html", "tool_input": {}},
            "execution_result": {"success": True, "type": "tool_execution", "content": "   ", "error": None},
        }
    )

    assert result["failure_type"] == "missing_source"
    legacy_key = "context" + "_evidence"
    assert legacy_key not in result


def test_reflection_returns_bad_artifact_for_failed_artifact_execution() -> None:
    result = reflection_node(
        {
            "phase": "artifact",
            "current_task": "task_1",
            "execution_result": {"success": False, "type": "tool_execution", "content": None, "error": "failed"},
        }
    )

    assert result["failure_type"] == "bad_artifact"
    legacy_key = "context" + "_evidence"
    assert legacy_key not in result


def test_reflection_updates_accomplishments_without_legacy_evidence_field() -> None:
    state = {
        "phase": "source",
        "agent_key": "assistant",
        "current_task": "task_1",
        "task_plan": [
            {
                "id": "task_1",
                "type": "source_gathering",
                "description": "Read source",
                "requires": ["source_loaded"],
                "context": {"tool": "browser_get_html"},
                "status": "pending",
            }
        ],
        "tool_call_history": [
            {
                "task_id": "task_1",
                "tool_name": "browser_get_html",
                "tool_input": {},
                "success": True,
                "execution_result": {"success": True, "type": "tool_execution", "content": {"text": "data"}, "error": None},
            }
        ],
        "decision": {"action": "use_tool", "tool_name": "browser_get_html", "tool_input": {}},
        "execution_result": {"success": True, "type": "tool_execution", "content": {"text": "data"}, "error": None},
    }

    result = reflection_node(state)
    assert result["failure_type"] is None
    assert "tool_call_history" in result
    legacy_key = "context" + "_evidence"
    assert legacy_key not in result
