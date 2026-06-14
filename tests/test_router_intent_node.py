import importlib

import pytest

intent_router_module = importlib.import_module("solidcue.core.graph_router.nodes.intent_router_node")
from solidcue.core.graph_router.nodes.intent_router_node import intent_router_node


class _RouterProvider:
    model = "router-stream-model"

    def __init__(self, output_json: str) -> None:
        self._output_json = output_json

    def generate(self, _messages, **_kwargs):
        return self._output_json


@pytest.mark.asyncio
async def test_intent_router_returns_json_payload(monkeypatch) -> None:
    provider = _RouterProvider(
        output_json=(
            '{"assistant_draft":"I can help with that.","router_intent":"clarify","router_next":"final_output","target_agent_key":"","route_reason":"LLM response","handoff":{}}'
        )
    )

    monkeypatch.setattr(
        intent_router_module,
        "get_runtime_router_provider",
        lambda _thread_id: provider,
    )

    result = await intent_router_node(
        {
            "thread_id": "thread-1",
            "user_input": "can you generate a resume for https://www.linkedin.com/jobs/view/4416496575 ?",
            "chat_history": [],
        },
    )

    assert result["router_intent"] == "clarify"
    assert result["router_next"] == "final_output"
    assert result["final_response"] == "I can help with that."
    assert result["assistant_draft"] == "I can help with that."
    assert result["handoff"] == {}


@pytest.mark.asyncio
async def test_intent_router_keeps_direct_instruction_as_task_when_provider_missing() -> None:
    result = await intent_router_node(
        {
            "thread_id": "thread-2",
            "user_input": "generate a resume for https://www.linkedin.com/jobs/view/4416496575",
            "chat_history": [],
        }
    )

    assert result["router_intent"] == "clarify"
    assert "provider" in str(result["final_response"]).casefold()


@pytest.mark.asyncio
async def test_intent_router_passes_through_task_handoff_from_llm(monkeypatch) -> None:
    provider = _RouterProvider(
        output_json=(
            '{"assistant_draft":"I\'ll route this to the JD Archiver.","router_intent":"task","router_next":"handoff","target_agent_key":"jd_archiver","route_reason":"Task requested","handoff":{"action":"route_agent","task_input":"archive this job","target_agent_key":"jd_archiver"}}'
        ),
    )
    monkeypatch.setattr(
        intent_router_module,
        "get_runtime_router_provider",
        lambda _thread_id: provider,
    )

    result = await intent_router_node(
        {
            "thread_id": "thread-3",
            "user_input": "archive this job",
            "chat_history": [],
        }
    )

    assert result["router_intent"] == "task"
    assert result["router_next"] == "handoff"
    assert result["target_agent_key"] == "jd_archiver"
    assert result["handoff"]["action"] == "route_agent"
    assert result["assistant_draft"] == "I'll route this to the JD Archiver."
