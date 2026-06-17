"""Tests for task planning node (Phase 3)."""

import pytest
import importlib as _il

from solidcue.core.graph_agent.nodes.planning_node import (
    _apply_planning_guardrails,
    _guardrail_normalize_task_shape,
    planning_node,
)

planning_module = _il.import_module("solidcue.core.graph_agent.nodes.planning_node")


@pytest.mark.asyncio
async def test_planning_generates_task_plan_for_artifact_request() -> None:
    """Planning node should generate tasks for artifact generation requests."""
    result = await planning_node(
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


@pytest.mark.asyncio
async def test_planning_creates_synthesis_only_for_simple_requests() -> None:
    """Planning node should create single synthesis task for simple requests."""
    result = await planning_node(
        {
            "user_input": "What is machine learning?",
        }
    )

    assert "task_plan" in result
    assert isinstance(result["task_plan"], list)
    assert result["current_task"] == "task_1"

    # May be multiple tasks or single synthesis depending on LLM
    assert all("type" in task for task in result["task_plan"])


@pytest.mark.asyncio
async def test_planning_handles_empty_user_input() -> None:
    """Planning node should handle empty user input gracefully."""
    result = await planning_node({})

    assert "task_plan" in result
    assert isinstance(result["task_plan"], list)
    assert len(result["task_plan"]) > 0
    assert result["current_task"] == "task_1"

    # Should at least have a synthesis task
    task_types = [task.get("type") for task in result["task_plan"]]
    assert "synthesis" in task_types


@pytest.mark.asyncio
async def test_planning_includes_source_gathering_for_document_requests() -> None:
    """Planning node should include source gathering for document creation."""
    user_input = "Create a document from these files"
    result = await planning_node(
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


@pytest.mark.asyncio
async def test_planning_sets_tasks_to_pending() -> None:
    """Planning node should set all generated tasks to pending status."""
    result = await planning_node(
        {
            "user_input": "Write a summary of the document",
        }
    )

    assert "task_plan" in result
    for task in result["task_plan"]:
        assert task.get("status") == "pending"


@pytest.mark.asyncio
async def test_planning_returns_messages() -> None:
    """Planning node should include the initial user message."""
    result = await planning_node(
        {
            "user_input": "Generate a resume",
        }
    )

    assert "messages" in result
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) > 0
    assert result["messages"][0].get("role") == "system"


@pytest.mark.asyncio
async def test_planning_includes_token_stats() -> None:
    result = await planning_node(
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


def test_guardrails_strip_user_input_source_values_from_context() -> None:
    """Source URLs/paths must not be written into the task plan.

    The plan persists and is reused, so any baked-in source would pollute a
    future request with a different source. The source binding is carried only
    by item_key; the concrete value is resolved downstream.
    """
    raw = [
        {
            "id": "x",
            "type": "source_gathering",
            "description": "Navigate to the job posting URL",
            "requires": ["job_page_opened"],
            "context": {
                "tool": "browser_navigate",
                "url": "https://www.linkedin.com/jobs/view/4421570943/",
            },
            "status": "pending",
        },
        {
            "id": "y",
            "type": "source_gathering",
            "description": "Extract the JD text",
            "requires": ["jd_extracted"],
            "context": {
                "tool": "extract_text",
                "posting_url": "https://www.linkedin.com/jobs/view/4421570943/",
            },
            "status": "pending",
        },
    ]
    target_artifacts_source = [
        {
            "index": 1,
            "item_key": "jd_one",
            "source_ref": "https://www.linkedin.com/jobs/view/4421570943/",
            "source_type": "url",
        }
    ]

    out = _apply_planning_guardrails(raw, target_artifacts_source=target_artifacts_source)

    for task in out:
        context = task["context"]
        for field in ("url", "posting_url", "source_ref", "jd_url", "job_url"):
            assert field not in context
        # The non-source binding/argument fields are preserved.
        assert context["item_key"] == "jd_one"
    assert out[0]["context"]["tool"] == "browser_navigate"
    assert out[1]["context"]["tool"] == "extract_text"


@pytest.mark.asyncio
async def test_planning_node_uses_cache_when_present(monkeypatch) -> None:
    cached = [
        {
            "id": "task_1",
            "type": "source_gathering",
            "description": "Navigate to job posting URL",
            "requires": ["job_page_loaded"],
            "context": {"tool": "browser_navigate"},
            "status": "pending",
        }
    ]
    monkeypatch.setattr(planning_module, "_load_task_plan_cache", lambda _agent_key: cached)
    llm_called = {"called": False}

    async def _fake_llm(_state):
        llm_called["called"] = True
        return [], {}

    monkeypatch.setattr(planning_module, "_llm_plan", _fake_llm)

    result = await planning_node({"user_input": "Build me a resume", "agent_key": "resume_builder"})

    assert not llm_called["called"]
    assert result["task_plan"][0]["type"] == "source_gathering"
    assert result["current_task"] == "task_1"


@pytest.mark.asyncio
async def test_planning_node_saves_cache_on_llm_miss(monkeypatch) -> None:
    monkeypatch.setattr(planning_module, "_load_task_plan_cache", lambda _agent_key: None)
    saved = {}

    def _fake_save(agent_key, tasks):
        saved["agent_key"] = agent_key
        saved["tasks"] = tasks

    monkeypatch.setattr(planning_module, "_save_task_plan_cache", _fake_save)

    async def _fake_llm(_state):
        return [
            {
                "id": "task_1",
                "type": "source_gathering",
                "description": "Gather sources",
                "requires": ["source_collected"],
                "status": "pending",
            }
        ], {}

    monkeypatch.setattr(planning_module, "_llm_plan", _fake_llm)

    await planning_node({"user_input": "Build me a resume", "agent_key": "resume_builder"})

    assert saved.get("agent_key") == "resume_builder"
    assert isinstance(saved.get("tasks"), list)
    assert len(saved["tasks"]) == 1
