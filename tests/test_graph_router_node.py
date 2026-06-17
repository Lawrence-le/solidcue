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
    assert "Previous tool call failed: Google API failed: HTTP 404 File not found" in retry_reason


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


def test_router_keeps_source_gathering_when_accomplishments_incomplete() -> None:
    """Router should stay in source phase when required accomplishments are not yet met."""
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
        # No tool_call_history with accomplishments — task is incomplete.
        "tool_call_history": [],
    }

    result = router_node(state)

    assert result["phase"] == "source"
    assert result["current_task"] == "task_1"
    assert result["router_next"] == "decision"


def test_router_advances_to_next_task_when_accomplishments_complete() -> None:
    """Router should advance current_task when all required accomplishments are met."""
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
        "tool_call_history": [
            {
                "task_id": "task_1",
                "tool_name": "drive_read_file",
                "tool_input": {"path": "profile.md"},
                "success": True,
                "accomplishments": ["profile_data_met"],
            }
        ],
    }

    result = router_node(state)

    assert result["current_task"] == "task_2"


def test_router_source_only_plan_finalizes_after_validated_synthesis() -> None:
    """A plan whose last task is source_gathering (no synthesis task, e.g. weather
    assistant) must reach final_output once a validated draft exists, not loop synthesis."""
    completed_source_plan = [
        {
            "id": "task_1",
            "type": "source_gathering",
            "description": "Retrieve current weather",
            "requires": ["weather_data_retrieved"],
            "status": "completed",
        }
    ]

    # First pass after source completes, before synthesis has run: route to synthesis.
    pre_synthesis = router_node(
        {
            "phase": "synthesis",
            "failure_type": None,
            "current_task": "task_1",
            "task_plan": completed_source_plan,
            "agent_key": "weather_assistant",
            "synthesis_draft": None,
        }
    )
    assert pre_synthesis["router_next"] == "synthesis"

    # After synthesis produced a validated draft (failure_type cleared): finalize.
    post_synthesis = router_node(
        {
            "phase": "synthesis",
            "failure_type": None,
            "current_task": "task_1",
            "task_plan": completed_source_plan,
            "agent_key": "weather_assistant",
            "synthesis_draft": "It is 22°C and sunny.",
        }
    )
    assert post_synthesis["router_next"] == "final_output"
    assert post_synthesis["phase"] == "final"


def test_router_artifact_completion_uses_task_scoped_tool_history() -> None:
    """Router should mark artifact phase final when all accomplishments for current task are met."""
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
                "requires": ["formatted_document", "upload_confirmation"],
                "status": "pending",
            }
        ],
        "tool_call_history": [
            # task_3 entry should NOT count toward task_4 completion
            {
                "task_id": "task_3",
                "tool_name": "create_formatted_word_document_base64",
                "tool_input": {"content": "x"},
                "success": True,
                "accomplishments": ["formatted_document_met"],
            },
            # task_4 entries supply the required accomplishments
            {
                "task_id": "task_4",
                "tool_name": "create_formatted_word_document_base64",
                "tool_input": {"content": "x"},
                "success": True,
                "accomplishments": ["formatted_document_met"],
            },
            {
                "task_id": "task_4",
                "tool_name": "drive_upload_file",
                "tool_input": {"name": "resume.docx"},
                "success": True,
                "accomplishments": ["upload_confirmation_met"],
            },
        ],
    }

    result = router_node(state)

    assert result["phase"] == "final"
    assert result["router_next"] == "final_output"


def test_router_skips_accomplishment_check_when_task_already_completed() -> None:
    """Router should advance immediately when task status is 'completed' without checking accomplishments."""
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
                "status": "completed",  # already done — no accomplishments needed
            },
            {
                "id": "task_2",
                "type": "synthesis",
                "description": "Draft response",
                "requires": ["complete_answer"],
                "status": "pending",
            },
        ],
        "tool_call_history": [],  # deliberately empty — status=completed bypasses check
    }

    result = router_node(state)

    assert result["current_task"] == "task_2"
    assert result["router_next"] == "synthesis"
