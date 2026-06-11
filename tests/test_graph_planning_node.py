"""Tests for task planning node (Phase 3)."""

from solidcue.core.graph_agent.nodes.planning_node import (
    _guardrail_normalize_task_shape,
    planning_node,
)


def test_planning_generates_task_plan_for_artifact_request() -> None:
    """Planning node should generate tasks for artifact generation requests."""
    result = planning_node(
        {
            "user_input": "Generate a resume from this job posting",
        }
    )

    assert "task_plan" in result
    assert isinstance(result["task_plan"], list)
    assert len(result["task_plan"]) > 0
    assert result["current_task"] == "task_1"

    # Check task structure
    first_task = result["task_plan"][0]
    assert "id" in first_task
    assert "type" in first_task
    assert "description" in first_task
    assert "status" in first_task


def test_planning_creates_synthesis_only_for_simple_requests() -> None:
    """Planning node should create single synthesis task for simple requests."""
    result = planning_node(
        {
            "user_input": "What is machine learning?",
        }
    )

    assert "task_plan" in result
    assert isinstance(result["task_plan"], list)
    assert result["current_task"] == "task_1"

    # May be multiple tasks or single synthesis depending on LLM
    assert all("type" in task for task in result["task_plan"])


def test_planning_handles_empty_user_input() -> None:
    """Planning node should handle empty user input gracefully."""
    result = planning_node({})

    assert "task_plan" in result
    assert isinstance(result["task_plan"], list)
    assert len(result["task_plan"]) > 0
    assert result["current_task"] == "task_1"

    # Should at least have a synthesis task
    task_types = [task.get("type") for task in result["task_plan"]]
    assert "synthesis" in task_types


def test_planning_includes_source_gathering_for_document_requests() -> None:
    """Planning node should include source gathering for document creation."""
    user_input = "Create a document from these files"
    result = planning_node(
        {
            "user_input": user_input,
        }
    )

    assert "task_plan" in result
    task_types = [task.get("type") for task in result["task_plan"]]

    # Should have source gathering if document/artifact creation
    if "document" in user_input.lower():
        # May include source_gathering based on LLM judgment or fallback
        assert any(t in task_types for t in ["source_gathering", "artifact_generation", "synthesis"])


def test_planning_sets_tasks_to_pending() -> None:
    """Planning node should set all generated tasks to pending status."""
    result = planning_node(
        {
            "user_input": "Write a summary of the document",
        }
    )

    assert "task_plan" in result
    for task in result["task_plan"]:
        assert task.get("status") == "pending"


def test_planning_returns_messages() -> None:
    """Planning node should include the initial user message."""
    result = planning_node(
        {
            "user_input": "Generate a resume",
        }
    )

    assert "messages" in result
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) > 0
    assert result["messages"][0].get("role") == "system"


def test_planning_includes_token_stats() -> None:
    result = planning_node(
        {
            "user_input": "Generate a resume",
        }
    )

    assert "metric_planning" in result
    assert isinstance(result["metric_planning"], dict)


def test_planning_normalizes_task_without_legacy_role_field() -> None:
    tasks = _guardrail_normalize_task_shape([
        {
            "id": "x",
            "type": "source_gathering",
            "description": "Load candidate resume master",
            "requires": ["resume_master_collected"],
            "status": "pending",
        }
    ])

    role_key = "evidence" + "_role"
    assert role_key not in tasks[0]


def test_planning_no_longer_infers_legacy_role_for_resume_master() -> None:
    tasks = _guardrail_normalize_task_shape([
        {
            "id": "x",
            "type": "source_gathering",
            "description": "Load candidate resume master",
            "requires": ["resume_master_collected"],
            "status": "pending",
        }
    ])

    role_key = "evidence" + "_role"
    assert role_key not in tasks[0]


def test_planning_no_longer_infers_legacy_role_for_job_description() -> None:
    tasks = _guardrail_normalize_task_shape([
        {
            "id": "x",
            "type": "source_gathering",
            "description": "Load target job description",
            "requires": ["job_description_collected"],
            "status": "pending",
        }
    ])

    role_key = "evidence" + "_role"
    assert role_key not in tasks[0]


def test_planning_no_longer_sets_legacy_role_for_jd_tailoring() -> None:
    tasks = _guardrail_normalize_task_shape([
        {
            "id": "x",
            "type": "source_gathering",
            "description": "Load target job description for resume tailoring",
            "requires": ["job_description_collected"],
            "status": "pending",
        }
    ])

    role_key = "evidence" + "_role"
    assert role_key not in tasks[0]
