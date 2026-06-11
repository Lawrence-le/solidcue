from unittest.mock import patch

from solidcue.core.graph_agent.nodes.router_node import router_node


def test_router_routes_missing_source_to_decision() -> None:
    result = router_node({"phase": "source", "failure_type": "missing_source"})

    assert result["router_next"] == "decision"
    assert result["phase"] == "source"
    assert result["source_attempt"] == 1


def test_router_increments_source_attempt_on_repeat_missing_source() -> None:
    result = router_node({"phase": "source", "failure_type": "missing_source", "source_attempt": 2})

    assert result["source_attempt"] == 3


def test_router_missing_source_retry_reason_includes_latest_task_failure_detail() -> None:
    result = router_node(
        {
            "phase": "source",
            "failure_type": "missing_source",
            "current_task": "task_3",
            "task_plan": [
                {
                    "id": "task_3",
                    "type": "source_gathering",
                    "description": "Download resume source",
                    "requires": ["master_resume_content_downloaded"],
                    "status": "pending",
                }
            ],
            "tool_call_history": [
                {
                    "task_id": "task_3",
                    "tool_name": "drive_download_file",
                    "success": False,
                    "execution_result": {
                        "success": False,
                        "type": "tool_execution",
                        "content": None,
                        "error": "Google API failed: HTTP 404 File not found",
                    },
                }
            ],
        }
    )

    retry_reason = result.get("retry_reason", "")
    assert "MISSING_ACTION: Previous tool call failed: Google API failed: HTTP 404 File not found" in retry_reason


def test_router_missing_source_uses_current_task_failure_detail_only() -> None:
    result = router_node(
        {
            "phase": "source",
            "failure_type": "missing_source",
            "current_task": "task_2",
            "task_plan": [
                {
                    "id": "task_2",
                    "type": "source_gathering",
                    "description": "Collect profile doc",
                    "requires": ["profile_doc_downloaded"],
                    "status": "pending",
                }
            ],
            "tool_call_history": [
                {
                    "task_id": "task_1",
                    "tool_name": "drive_download_file",
                    "success": False,
                    "execution_result": {
                        "success": False,
                        "type": "tool_execution",
                        "content": None,
                        "error": "Task 1 old failure",
                    },
                },
                {
                    "task_id": "task_2",
                    "tool_name": "drive_download_file",
                    "success": False,
                    "execution_result": {
                        "success": False,
                        "type": "tool_execution",
                        "content": None,
                        "error": "Task 2 current failure",
                    },
                },
            ],
        }
    )

    retry_reason = result.get("retry_reason", "")
    assert "Task 2 current failure" in retry_reason
    assert "Task 1 old failure" not in retry_reason


def test_router_routes_artifact_required_retry_to_pending_artifact_task() -> None:
    result = router_node(
        {
            "phase": "synthesis",
            "failure_type": "missing_source",
            "validation_report": {
                "reason": "ARTIFACT_REQUIRED: The user requested a generated resume document.",
                "score": 0.2,
            },
            "current_task": "task_2",
            "task_plan": [
                {
                    "id": "task_1",
                    "type": "source_gathering",
                    "description": "Gather source context",
                    "status": "pending",
                },
                {
                    "id": "task_2",
                    "type": "synthesis",
                    "description": "Draft resume content",
                    "status": "pending",
                },
                {
                    "id": "task_3",
                    "type": "artifact_generation",
                    "description": "Create resume document",
                    "status": "pending",
                },
            ],
        }
    )

    assert result["phase"] == "artifact"
    assert result["current_task"] == "task_3"
    assert result["router_next"] == "decision"
    assert result["failure_type"] is None


def test_router_routes_artifact_required_retry_without_task_plan_to_artifact_phase() -> None:
    result = router_node(
        {
            "phase": "synthesis",
            "failure_type": "missing_source",
            "user_input": "Generate a resume document",
            "validation_report": {
                "reason": "ARTIFACT_REQUIRED: The requested document has not been created.",
                "score": 0.2,
            },
        }
    )

    assert result["phase"] == "artifact"
    assert result["router_next"] == "decision"
    assert result["failure_type"] is None


def test_router_routes_bad_artifact_to_artifact_generation() -> None:
    result = router_node({"phase": "artifact", "failure_type": "bad_artifact"})

    assert result["router_next"] == "decision"
    assert result["phase"] == "artifact"
    assert result["artifact_attempt"] == 1


def test_router_routes_not_executed_to_artifact_generation() -> None:
    result = router_node({"phase": "artifact", "failure_type": "not_executed"})

    assert result["router_next"] == "decision"
    assert result["phase"] == "artifact"
    assert result["artifact_attempt"] == 1


def test_router_routes_bad_synthesis_to_synthesis() -> None:
    result = router_node({"phase": "synthesis", "failure_type": "bad_synthesis"})

    assert result["router_next"] == "synthesis"
    assert result["phase"] == "synthesis"
    assert result["synthesis_attempt"] == 1


def test_router_transitions_source_to_artifact_when_intent_is_artifact() -> None:
    result = router_node(
        {
            "phase": "source",
            "failure_type": None,
            "user_input": "Generate a resume PDF for the role",
        }
    )

    assert result["phase"] == "artifact"
    assert result["router_next"] == "decision"
    assert result["failure_type"] is None


def test_router_transitions_source_to_synthesis_when_no_artifact_intent() -> None:
    result = router_node(
        {
            "phase": "source",
            "failure_type": None,
            "user_input": "Summarize what you found",
        }
    )

    assert result["phase"] == "synthesis"
    assert result["router_next"] == "synthesis"
    assert result["failure_type"] is None


def test_router_routes_artifact_phase_with_no_result_to_artifact_generation() -> None:
    result = router_node({"phase": "artifact", "failure_type": None})

    assert result["phase"] == "artifact"
    assert result["router_next"] == "decision"


def test_router_routes_artifact_phase_with_result_to_synthesis() -> None:
    result = router_node(
        {
            "phase": "artifact",
            "failure_type": None,
            "decision": {"action": "use_tool", "tool_name": "create_formatted_word_document_base64"},
            "execution_result": {"success": True, "content": {"fileId": "file_123"}},
        }
    )

    assert result["phase"] == "synthesis"
    assert result["router_next"] == "synthesis"


def test_router_routes_synthesis_phase_to_final_output() -> None:
    result = router_node({"phase": "synthesis", "failure_type": None})

    assert result["phase"] == "final"
    assert result["router_next"] == "final_output"


def test_router_terminates_when_retry_limit_reached() -> None:
    result = router_node(
        {
            "phase": "source",
            "failure_type": "missing_source",
            "max_retries": 5,
            "source_attempt": 3,
            "artifact_attempt": 1,
            "synthesis_attempt": 1,
        }
    )

    assert result["phase"] == "final"
    assert result["failure_type"] == "retry_limit"
    assert result["router_next"] == "final_output"


def test_router_does_not_inspect_router_origin_or_reflection_result() -> None:
    """Regression: router should ignore legacy router_origin/reflection_result keys."""
    result = router_node(
        {
            "phase": "source",
            "failure_type": "missing_source",
            "router_origin": "reflection",
            "reflection_result": {"sufficient": True, "reason": "x", "missing": None},
        }
    )

    assert result["router_next"] == "decision"
    assert result["source_attempt"] == 1


def test_router_keeps_source_gathering_when_llm_says_incomplete() -> None:
    """Router should stay in source phase when LLM task completion check returns False."""
    state = {
        "phase": "source",
        "failure_type": None,
        "current_task": "task_1",
        "task_plan": [
            {
                "id": "task_1",
                "type": "source_gathering",
                "description": "Gather sources",
                "requires": ["profile_data", "experience_data"],
                "status": "pending",
            }
        ],
        "agent_key": "resume_builder",
    }

    # LLM says task is not yet complete
    with patch("solidcue.core.graph_agent.nodes.router_node._llm_task_complete", return_value=(False, [], ["profile_data", "experience_data"], "Read the listed files using drive_read_file.")):
        result = router_node(state)

    assert result["phase"] == "source"
    assert result["current_task"] == "task_1"
    assert result["router_next"] == "decision"


def test_router_advances_to_next_task_when_llm_says_complete() -> None:
    """Router should advance current_task and mark prior task completed when LLM returns True."""
    state = {
        "phase": "source",
        "failure_type": None,
        "current_task": "task_1",
        "task_plan": [
            {
                "id": "task_1",
                "type": "source_gathering",
                "description": "Gather sources",
                "requires": ["profile_data"],
                "status": "pending",
            },
            {
                "id": "task_2",
                "type": "synthesis",
                "description": "Draft response",
                "requires": ["complete_answer"],
                "status": "pending",
            },
        ],
        "agent_key": "resume_builder",
    }

    with patch("solidcue.core.graph_agent.nodes.router_node._llm_task_complete", return_value=(True, ["profile_data"], [], "")):
        result = router_node(state)

    assert result["current_task"] == "task_2"


def test_router_artifact_completion_uses_task_scoped_tool_history() -> None:
    state = {
        "phase": "artifact",
        "failure_type": None,
        "agent_key": "resume_builder",
        "current_task": "task_4",
        "task_plan": [
            {
                "id": "task_4",
                "type": "artifact_generation",
                "description": "Create and upload resume doc",
                "requires": ["formatted document", "upload confirmation"],
                "status": "pending",
            }
        ],
        "tool_call_history": [
            {
                "task_id": "task_3",
                "tool_name": "create_formatted_word_document_base64",
                "tool_input": {"content": "x"},
                "success": True,
                "execution_result": {
                    "success": True,
                    "type": "tool_execution",
                    "content": {"content_base64": "abc"},
                    "error": None,
                },
            },
            {
                "task_id": "task_4",
                "tool_name": "drive_upload_file",
                "tool_input": {"name": "resume.docx"},
                "success": True,
                "execution_result": {
                    "success": True,
                    "type": "tool_execution",
                    "content": {"file_id": "f123", "webViewLink": "https://docs.google.com/..."},
                    "error": None,
                },
            },
        ],
    }

    with patch(
        "solidcue.core.graph_agent.nodes.router_node._llm_artifact_task_complete",
        return_value=(True, ["formatted document", "upload confirmation"], [], ""),
    ):
        result = router_node(state)

    assert result["phase"] == "final"
    assert result["router_next"] == "final_output"


def test_router_skips_llm_check_when_task_already_completed() -> None:
    """Router should skip LLM call and advance immediately when task status is 'completed'."""
    state = {
        "phase": "source",
        "failure_type": None,
        "current_task": "task_1",
        "task_plan": [
            {
                "id": "task_1",
                "type": "source_gathering",
                "description": "Gather sources",
                "requires": ["profile_data"],
                "status": "completed",  # already done
            },
            {
                "id": "task_2",
                "type": "synthesis",
                "description": "Draft response",
                "requires": ["complete_answer"],
                "status": "pending",
            },
        ],
    }

    with patch("solidcue.core.graph_agent.nodes.router_node._llm_task_complete") as mock_llm:
        result = router_node(state)
        mock_llm.assert_not_called()

    assert result["current_task"] == "task_2"
    assert result["router_next"] == "synthesis"
