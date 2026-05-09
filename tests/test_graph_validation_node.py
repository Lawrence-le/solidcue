from solidcue.core.graph_node import validation_node as validation_module
from solidcue.core.graph_node.validation_node import validation_node


def test_graph_validation_requests_artifact_retry_when_intent_needs_artifact(monkeypatch) -> None:
    monkeypatch.setattr(
        validation_module,
        "_llm_validate",
        lambda state, draft_output: {
            "passed": False,
            "reason": "ARTIFACT_REQUIRED: The user asked for a resume document, but no artifact was produced.",
            "score": 0.0,
            "retry_tag": "artifact_required",
        },
    )

    result = validation_node(
        {
            "user_input": "Generate a resume document from this job posting",
            "tool_call_history": [{"tool_name": "search_web", "tool_input": {"query": "job post"}}],
            "draft_output": "Here is a short summary only.",
            "attempt": 0,
        }
    )

    assert result["validation_result"]["passed"] is False
    assert result["retry_reason"].startswith("ARTIFACT_REQUIRED:")


def test_graph_validation_allows_non_artifact_path_to_continue(monkeypatch) -> None:
    monkeypatch.setattr(
        validation_module,
        "_llm_validate",
        lambda state, draft_output: {"passed": True, "reason": "ok", "score": 1.0, "retry_tag": "none"},
    )

    result = validation_node(
        {
            "user_input": "What is a queue in data structures?",
            "tool_call_history": [],
            "draft_output": "A queue is a FIFO data structure.",
            "attempt": 0,
        }
    )

    assert result["validation_result"]["passed"] is True
