from solidcue.core.graph_node import validation_llm_node as validation_module
from solidcue.core.graph_node.validation_llm_node import validation_llm_node as validation_node
from solidcue.prompts.validation_llm_system_prompt import build_validation_llm_system_prompt


def test_validation_prompt_checks_spelling() -> None:
    prompt = build_validation_llm_system_prompt()

    assert "spelling errors" in prompt
    assert "typos" in prompt


def test_validation_prompt_explains_validation_evidence() -> None:
    prompt = build_validation_llm_system_prompt()

    assert "validation_evidence" in prompt
    assert "does not fabricate details absent from evidence" in prompt


def test_graph_validation_emits_bad_synthesis_when_validator_fails_in_synthesis(monkeypatch) -> None:
    monkeypatch.setattr(
        validation_module,
        "_llm_validate",
        lambda state, draft_output: (
            {
                "passed": False,
                "reason": "The draft does not include required details from context.",
                "score": 0.0,
            },
            {},
        ),
    )

    result = validation_node(
        {
            "user_input": "Generate a resume document from this job posting",
            "tool_call_history": [{"tool_name": "search_web", "tool_input": {"query": "job post"}}],
            "synthesis_draft": "Here is a short summary only.",
        }
    )

    assert result["failure_type"] == "bad_synthesis"
    assert "required details" in result["validation_report"]["reason"].lower()
    assert "validation_result" not in result
    assert "retry_reason" not in result
    assert "router_origin" not in result
    assert "attempt" not in result


def test_graph_validation_emits_bad_synthesis_when_validator_fails_in_artifact_phase(monkeypatch) -> None:
    monkeypatch.setattr(
        validation_module,
        "_llm_validate",
        lambda state, draft_output: (
            {
                "passed": False,
                "reason": "Artifact content is incomplete.",
                "score": 0.2,
            },
            {},
        ),
    )

    result = validation_node(
        {
            "user_input": "Generate a resume",
            "phase": "artifact",
            "synthesis_draft": "partial draft",
        }
    )

    assert result["failure_type"] == "bad_synthesis"
    assert "incomplete" in result["validation_report"]["reason"].lower()


def test_graph_validation_emits_bad_synthesis_when_artifact_execution_failed() -> None:
    result = validation_node(
        {
            "user_input": "Generate a resume",
            "phase": "artifact",
            "artifact_result": {"success": False, "error": "tool timeout"},
        }
    )

    assert result["failure_type"] == "bad_synthesis"
    assert "draft output must be a string" in result["validation_report"]["reason"].lower()


def test_graph_validation_emits_null_failure_type_when_passed(monkeypatch) -> None:
    monkeypatch.setattr(
        validation_module,
        "_llm_validate",
        lambda state, draft_output: ({"passed": True, "reason": "ok", "score": 1.0}, {}),
    )

    result = validation_node(
        {
            "user_input": "What is a queue in data structures?",
            "tool_call_history": [],
            "synthesis_draft": "A queue is a FIFO data structure.",
        }
    )

    assert result["failure_type"] is None
    assert result["validation_report"]["score"] == 1.0
    assert "validation_result" not in result
    assert "finalization_reason" not in result


def test_graph_validation_emits_bad_synthesis_for_empty_draft() -> None:
    result = validation_node(
        {
            "user_input": "Tell me about queues",
            "synthesis_draft": "   ",
        }
    )

    assert result["failure_type"] == "bad_synthesis"
    assert "empty" in result["validation_report"]["reason"].lower()


def test_graph_validation_emits_bad_synthesis_for_control_token_leak() -> None:
    result = validation_node(
        {
            "user_input": "Tell me about queues",
            "synthesis_draft": "Here is the answer<|channel|>analysis<|message|>leak",
        }
    )

    assert result["failure_type"] == "bad_synthesis"
    assert "control tokens" in result["validation_report"]["reason"]


def test_graph_validation_includes_metric_validation_field(monkeypatch) -> None:
    monkeypatch.setattr(
        validation_module,
        "_llm_validate",
        lambda state, draft_output: (
            {"passed": True, "reason": "ok", "score": 1.0},
            {"estimated_total": 42, "estimated_system": 10, "estimated_user": 28, "message_count": 2},
        ),
    )

    result = validation_node(
        {
            "user_input": "Tell me about queues",
            "synthesis_draft": "A queue is FIFO.",
        }
    )

    assert "metric_validation" in result
    assert isinstance(result["metric_validation"], dict)


def test_llm_validation_uses_handoff_scoped_evidence(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(validation_module, "load_agent", lambda _: object())
    monkeypatch.setattr(validation_module, "get_provider_for_role", lambda _agent, _role: object())

    def fake_build_validation_messages(*, user_query, draft_output, validation_evidence, task_description=""):
        captured["validation_evidence"] = validation_evidence
        return [{"role": "user", "content": "validate"}]

    monkeypatch.setattr(validation_module, "build_validation_messages", fake_build_validation_messages)
    monkeypatch.setattr(
        validation_module,
        "timed_generate",
        lambda _provider, _messages, **_kwargs: ('{"passed": true, "reason": "ok", "score": 1.0}', {}),
    )

    result = validation_node(
        {
            "agent_key": "x",
            "user_input": "Create a resume",
            "synthesis_draft": "Draft",
            "task_plan": [{"id": "task_1", "context": {"item_key": "u_1"}}],
            "current_task": "task_1",
            "handoff": {
                "resume_text::u_1": {"content": "Resume master"},
                "jd_text::u_1": {"content": "Job description"},
                "global::folder": {"content": "Shared folder metadata"},
            },
        }
    )

    assert result["failure_type"] is None
    assert len(captured["validation_evidence"]) >= 2
