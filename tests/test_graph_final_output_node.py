from solidcue.core.graph_node.final_output_node import _compact_successful_tool_history, final_output_node


_DISALLOWED_FINAL_OUTPUT_WRITES = (
    "final_output",
    "workflow_status",
    "retry_reason",
    "draft_output",
    "finalization_reason",
    "phase",
    "failure_type",
    "synthesis_draft",
)


def test_final_output_writes_only_final_response_from_synthesis_draft() -> None:
    result = final_output_node(
        {
            "synthesis_draft": "Polished response from synthesis.",
        }
    )

    assert result["final_response"] == "Polished response from synthesis."
    assert "metric_final_output" in result
    assert isinstance(result["metric_final_output"], dict)
    for key in _DISALLOWED_FINAL_OUTPUT_WRITES:
        assert key not in result, f"final_output_node must not write '{key}'"


def test_final_output_ignores_legacy_draft_output_and_uses_fallback() -> None:
    result = final_output_node(
        {
            "draft_output": "Legacy draft output.",
        }
    )

    assert result["final_response"] == "I couldn't generate a final response for this request."
    assert "metric_final_output" in result


def test_final_output_uses_fallback_when_no_response_available() -> None:
    result = final_output_node({})

    assert result["final_response"] == "I couldn't generate a final response for this request."
    assert "metric_final_output" in result


def test_final_output_prefers_synthesis_draft_over_draft_output() -> None:
    result = final_output_node(
        {
            "synthesis_draft": "Synthesis version",
            "draft_output": "Draft version",
        }
    )

    assert result["final_response"] == "Synthesis version"


def test_final_output_uses_fallback_on_execution_failure() -> None:
    result = final_output_node(
        {
            "execution_result": {"success": False},
        }
    )

    assert "couldn't retrieve enough reliable information" in result["final_response"]


def test_compact_successful_tool_history_truncates_large_content() -> None:
    large_content = "x" * 5000
    history = _compact_successful_tool_history(
        {
            "tool_call_history": [
                {
                    "task_id": "task_1",
                    "tool_name": "search_web",
                    "tool_input": {"query": "q"},
                    "accomplishments": ["facts_met"],
                    "execution_result": {"success": True, "content": large_content},
                }
            ]
        }
    )

    assert len(history) == 1
    content = history[0]["content"]
    assert isinstance(content, str)
    assert len(content) < len(large_content)
    assert content.endswith("...[truncated]")
