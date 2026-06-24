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
        output_json='{"assistant_draft":"I can help with that.","router_intent":"clarify","route_reason":"LLM response"}'
    )
    monkeypatch.setattr(intent_router_module, "resolve_router_provider", lambda _thread_id: provider)

    result = await intent_router_node(
        {
            "thread_id": "thread-1",
            "user_input": "can you generate a resume for https://example.com/jobs/1 ?",
            "chat_history": [],
        },
    )

    assert result["router_intent"] == "clarify"
    assert result["router_next"] == "final_output"
    assert result["final_response"] == "I can help with that."
    assert result["assistant_draft"] == "I can help with that."
    # Classification-only: the intent router no longer emits a plan or handoff.
    assert "plan" not in result
    assert "handoff" not in result


@pytest.mark.asyncio
async def test_intent_router_returns_clarify_when_provider_invalid(monkeypatch) -> None:
    def _raise_missing(_thread_id):
        raise ValueError("No provider configured")

    monkeypatch.setattr(intent_router_module, "resolve_router_provider", _raise_missing)

    result = await intent_router_node(
        {
            "thread_id": "thread-2",
            "user_input": "generate a resume for https://example.com/jobs/1",
            "chat_history": [],
        }
    )

    assert result["router_intent"] == "clarify"
    assert result["router_next"] == "final_output"
    assert "provider" in str(result["final_response"]).casefold()


@pytest.mark.asyncio
async def test_intent_router_classifies_task_without_building_a_plan(monkeypatch) -> None:
    # The intent router only classifies; planning (plan/handoff/target_agent_key) is
    # now build_plan_node's responsibility.
    provider = _RouterProvider(
        output_json='{"assistant_draft":"On it.","router_intent":"task","route_reason":"Task requested"}'
    )
    monkeypatch.setattr(intent_router_module, "resolve_router_provider", lambda _thread_id: provider)

    result = await intent_router_node(
        {"thread_id": "thread-3", "user_input": "archive this job", "chat_history": []}
    )

    assert result["router_intent"] == "task"
    assert result["assistant_draft"] == "On it."
    assert "plan" not in result
    assert "handoff" not in result
    assert "target_agent_key" not in result
