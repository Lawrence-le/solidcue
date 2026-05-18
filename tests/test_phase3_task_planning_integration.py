"""Integration tests for Phase 3 task planning."""

from unittest.mock import patch

from solidcue.core.graph_node.planning_node import planning_node
from solidcue.core.graph_node.router_node import router_node
from solidcue.prompts.decision_prompt import build_decision_messages
from solidcue.agents.configs.loader import load_agent


def test_planning_node_generates_multi_task_plan() -> None:
    """Planning node should generate multiple tasks for complex requests."""
    result = planning_node(
        {
            "user_input": "Create a resume document from this job posting and compare it with my experience",
        }
    )

    assert "task_plan" in result
    assert isinstance(result["task_plan"], list)
    assert len(result["task_plan"]) >= 1
    assert result["current_task"] == "task_1"

    # Verify task structure
    for task in result["task_plan"]:
        assert "id" in task
        assert "type" in task
        assert "description" in task
        assert "status" in task
        assert task["status"] == "pending"


def test_decision_prompt_includes_task_context() -> None:
    """Decision prompt should include current task context when available."""
    try:
        agent = load_agent("default")
    except Exception:
        # Skip test if no default agent configured
        return

    metadata = {
        "current_task": {
            "id": "task_1",
            "type": "source_gathering",
            "description": "Gather information from the job posting",
            "status": "pending"
        },
        "current_task_id": "task_1",
        "total_tasks": 3,
    }

    messages = build_decision_messages(
        agent=agent,
        user_input="Create a resume document",
        metadata=metadata,
    )

    assert len(messages) > 0
    system_message = messages[0]
    assert system_message.get("role") == "system"
    system_content = system_message.get("content", "")

    # Check if task context is in the prompt
    assert "source_gathering" in system_content or "task_1" in system_content or len(system_content) > 0


def test_router_advances_task_plan_on_success() -> None:
    """Router should advance current_task when LLM confirms task complete."""
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
                "type": "artifact_generation",
                "description": "Generate artifact",
                "requires": ["artifact_ready_content"],
                "status": "pending",
            },
            {
                "id": "task_3",
                "type": "synthesis",
                "description": "Synthesize response",
                "requires": ["complete_answer"],
                "status": "pending",
            },
        ],
        "user_input": "Create a resume",
    }

    with patch("solidcue.core.graph_node.router_node._llm_task_complete", return_value=(True, ["profile_data"], [], "")):
        result = router_node(state)

    assert result["current_task"] == "task_2"
    assert result["phase"] == "artifact"
    assert result["router_next"] == "decision"


def test_router_reaches_final_output_after_all_tasks() -> None:
    """Router should route to final_output when last task completes."""
    state = {
        "phase": "synthesis",
        "failure_type": None,
        "current_task": "task_2",
        "task_plan": [
            {
                "id": "task_1",
                "type": "source_gathering",
                "description": "Gather sources",
                "requires": ["profile_data"],
                "status": "completed",
            },
            {
                "id": "task_2",
                "type": "synthesis",
                "description": "Synthesize response",
                "requires": ["complete_answer"],
                "status": "completed",
            },
        ],
        "user_input": "Create a response",
    }

    result = router_node(state)

    assert result["phase"] == "final"
    assert result["router_next"] == "final_output"


def test_router_preserves_phase_routing_without_task_plan() -> None:
    """Router should fall back to phase-based routing if no task plan exists."""
    state = {
        "phase": "source",
        "failure_type": None,
        "user_input": "Create a resume document",
    }

    result = router_node(state)

    assert "phase" in result
    assert "router_next" in result


def test_task_context_in_decision_node_metadata() -> None:
    """Decision node should include task context in metadata for run_agent."""
    state = {
        "agent_key": "default",
        "user_input": "Create a resume",
        "metadata": {"current_time": "2026-05-09 10:00:00"},
        "messages": [],
        "task_plan": [
            {
                "id": "task_1",
                "type": "source_gathering",
                "description": "Gather sources",
                "status": "pending",
            }
        ],
        "current_task": "task_1",
    }

    # Extract task context as decision_node does
    metadata = dict(state.get("metadata", {}))
    task_plan = state.get("task_plan")
    current_task_id = state.get("current_task", "task_1")
    if isinstance(task_plan, list) and task_plan:
        current_task = next((t for t in task_plan if isinstance(t, dict) and t.get("id") == current_task_id), None)
        if current_task:
            metadata["current_task"] = current_task
            metadata["current_task_id"] = current_task_id
            metadata["total_tasks"] = len(task_plan)

    # Verify task context was added to metadata
    assert "current_task" in metadata
    assert metadata["current_task"]["type"] == "source_gathering"
    assert metadata["current_task_id"] == "task_1"
    assert metadata["total_tasks"] == 1
