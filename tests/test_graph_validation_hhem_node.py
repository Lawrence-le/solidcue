from unittest.mock import patch

from solidcue.core.graph_agent.nodes.validation_hhem_node import (
    validation_hhem_node,
    _split_claims,
    _build_premise,
    _chunk_premise,
    _build_verifier_premise,
    _select_relevant_chunks,
    _score_groundedness,
)


def _make_state(draft=None, handoff=None, user_input=None, agent_key=None):
    state = {}
    if draft is not None:
        state["synthesis_draft"] = draft
    if handoff is not None:
        state["handoff"] = handoff
        state["task_plan"] = [{"id": "task_1", "context": {"item_key": "u_1"}}]
        state["current_task"] = "task_1"
    if user_input is not None:
        state["user_input"] = user_input
    if agent_key is not None:
        state["agent_key"] = agent_key
    return state


def test_empty_draft_fails():
    result = validation_hhem_node(_make_state(draft=""))
    assert result["failure_type"] == "bad_synthesis"
    assert "empty" in result["validation_report"]["reason"].lower()


def test_none_draft_fails():
    result = validation_hhem_node(_make_state(draft=None))
    assert result["failure_type"] == "bad_synthesis"


def test_no_premise_skips_with_pass():
    result = validation_hhem_node(_make_state(draft="Some output", handoff={}, user_input=""))
    assert result["failure_type"] is None
    assert result["validation_report"]["score"] == 1.0


@patch("solidcue.core.graph_agent.nodes.validation_hhem_node._score_groundedness")
def test_all_claims_grounded_passes(mock_score):
    mock_score.return_value = (0.85, [{"claim": "Python is a language.", "score": 0.85}], {})
    result = validation_hhem_node(_make_state(
        draft="Python is a programming language.",
        handoff={"source::u_1": {"content": "Python is a popular programming language."}},
    ))
    assert result["failure_type"] is None
    assert result["validation_report"]["score"] == 0.85


@patch("solidcue.core.graph_agent.nodes.validation_hhem_node._llm_verify_failures")
@patch("solidcue.core.graph_agent.nodes.validation_hhem_node._score_groundedness")
def test_hhem_fails_but_llm_clears_metadata(mock_score, mock_llm):
    mock_score.return_value = (0.01, [
        {"claim": "# Darren Liew", "score": 0.01},
        {"claim": "Led a team of 4 engineers.", "score": 0.85},
    ], {})
    mock_llm.return_value = ([], "All flagged items are metadata.", {})
    result = validation_hhem_node(_make_state(
        draft="# Darren Liew\nLed a team of 4 engineers.",
        handoff={"source::u_1": {"content": "Darren led engineering teams."}},
        agent_key="test_agent",
    ))
    assert result["failure_type"] is None
    assert "metadata" in result["validation_report"]["reason"].lower()


@patch("solidcue.core.graph_agent.nodes.validation_hhem_node._llm_verify_failures")
@patch("solidcue.core.graph_agent.nodes.validation_hhem_node._score_groundedness")
def test_hhem_fails_and_llm_confirms_hallucination(mock_score, mock_llm):
    mock_score.return_value = (0.1, [
        {"claim": "Managed a $2M budget annually.", "score": 0.1},
    ], {})
    mock_llm.return_value = (["Managed a $2M budget annually."], "Not supported by source.", {})
    result = validation_hhem_node(_make_state(
        draft="Managed a $2M budget annually.",
        handoff={"source::u_1": {"content": "Junior developer for 2 years."}},
        agent_key="test_agent",
    ))
    assert result["failure_type"] == "bad_synthesis"
    assert "Managed a $2M budget" in result["validation_report"]["reason"]


@patch("solidcue.core.graph_agent.nodes.validation_hhem_node._score_groundedness")
def test_no_premise_from_handoff_skips_scoring(mock_score):
    mock_score.return_value = (0.7, [{"claim": "The answer is 42.", "score": 0.7}], {})
    result = validation_hhem_node(_make_state(
        draft="The answer is 42.",
        user_input="What is the meaning of life?",
    ))
    mock_score.assert_not_called()
    assert result["failure_type"] is None


def test_build_premise_extracts_handoff_text():
    premise = _build_premise({
        "handoff": {
            "alignment::u_1": {"content": "Job requires Kubernetes."},
            "grounding::u_1": {"content": "Candidate used Python."},
            "global::ctx": {"content": "Company background."},
        },
        "task_plan": [{"id": "task_1", "context": {"item_key": "u_1"}}],
        "current_task": "task_1",
    })

    assert "Candidate used Python" in premise
    assert "Job requires Kubernetes" in premise
    assert "Company background" in premise


def test_build_premise_extracts_nested_content_text():
    premise = _build_premise({
        "handoff": {
            "master_resume_downloaded::u_1": {
                "name": "resume_master",
                "content": "Candidate resume master full text.",
                "mimeType": "text/plain",
            }
        },
        "task_plan": [{"id": "task_1", "context": {"item_key": "u_1"}}],
        "current_task": "task_1",
    })

    assert premise == "Candidate resume master full text."


def test_build_premise_returns_empty_when_no_handoff():
    premise = _build_premise({
        "handoff": {}
    })

    assert premise == ""


def test_chunk_premise_keeps_long_single_paragraph_content():
    premise = "A" * 1300 + "B" * 1300
    chunks = _chunk_premise(premise)

    assert len(chunks) >= 2
    assert "B" * 100 in "".join(chunks)


def test_relevant_chunk_selection_can_find_late_resume_terms():
    chunks = [
        "Professional summary with Python and APIs.",
        "Earlier experience with Kubernetes and CI/CD.",
        "Technical Skills include Redis, SQLite, TypeScript, and Tailwind CSS.",
    ]

    selected = _select_relevant_chunks("Redis SQLite", chunks)

    assert any("Redis, SQLite" in chunk for chunk in selected)


def test_verifier_premise_uses_relevant_late_chunks():
    premise = (
        ("Earlier unrelated content about project delivery.\n\n" * 20)
        + "Technical Skills include Redis, SQLite, TypeScript, and Tailwind CSS."
    )

    verifier_premise = _build_verifier_premise(
        premise,
        [{"claim": "Redis"}, {"claim": "SQLite"}],
    )

    assert "Redis" in verifier_premise
    assert "SQLite" in verifier_premise


def test_score_groundedness_scores_all_premise_chunks(monkeypatch):
    class FakeModel:
        def predict(self, pairs):
            return [0.9 if "late grounding" in pair[0] else 0.1 for pair in pairs]

    monkeypatch.setattr(
        "solidcue.core.graph_agent.nodes.validation_hhem_node.get_hhem_model",
        lambda: FakeModel(),
    )
    premise = ("early unrelated content.\n\n" * 20) + "late grounding evidence supports Redis."

    min_score, claim_scores, stats = _score_groundedness(premise, "Redis is used.")

    assert min_score == 0.9
    assert claim_scores[0]["score"] == 0.9
    assert stats["hhem_pair_count"] == stats["hhem_chunk_count"]


def test_llm_verify_uses_max_tokens_cap(monkeypatch):
    captured = {}

    class Provider:
        model = "test-model"

    monkeypatch.setattr(
        "solidcue.core.graph_agent.nodes.validation_hhem_node.load_agent",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "solidcue.core.graph_agent.nodes.validation_hhem_node.get_provider_for_role",
        lambda _agent, _role: Provider(),
    )

    def fake_timed_generate(provider, messages, *, node_name="llm", max_tokens=None):
        captured["max_tokens"] = max_tokens
        captured["node_name"] = node_name
        return '{"real_failures": [], "reason": "ok"}', {"tokens": {}, "time_s": 0.0, "model": "test-model"}

    monkeypatch.setattr(
        "solidcue.core.graph_agent.nodes.validation_hhem_node.timed_generate",
        fake_timed_generate,
    )

    from solidcue.core.graph_agent.nodes.validation_hhem_node import _llm_verify_failures

    real_failures, _, _ = _llm_verify_failures(
        {"agent_key": "test_agent"},
        [{"claim": "Redis", "score": 0.1}],
        "Resume premise",
    )

    assert real_failures == []
    assert captured["node_name"] == "validation_hhem"
    assert captured["max_tokens"] == 300


def test_split_claims_by_sentences():
    text = "Led a team of 4 engineers. Deployed AWS infrastructure."
    claims = _split_claims(text)
    assert len(claims) == 2
    assert "Led a team" in claims[0]
    assert "Deployed AWS" in claims[1]


def test_split_claims_by_bullets():
    text = "• Led a team of 4 engineers\n• Deployed AWS infrastructure\n• Managed $2M budget"
    claims = _split_claims(text)
    assert len(claims) == 3


def test_split_claims_skips_short_fragments():
    text = "OK. This is a real sentence with substance."
    claims = _split_claims(text)
    assert len(claims) == 1
    assert "real sentence" in claims[0]
