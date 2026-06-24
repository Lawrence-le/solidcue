import importlib

import pytest

build_plan_module = importlib.import_module("solidcue.core.graph_router.nodes.build_plan_node")
from solidcue.core.graph_router.nodes.build_plan_node import build_plan_node


class _PlanProvider:
    model = "plan-model"

    def __init__(self, output_json: str) -> None:
        self._output_json = output_json

    def generate(self, _messages, **_kwargs):
        return self._output_json


def _real_agent_keys() -> set[str]:
    return {a["agent_key"] for a in build_plan_module.available_agents()}


@pytest.mark.asyncio
async def test_build_plan_emits_plan_for_valid_agents(monkeypatch) -> None:
    keys = _real_agent_keys()
    if "weather_assistant" not in keys:
        pytest.skip("weather_assistant agent not registered in this environment")

    provider = _PlanProvider(
        '{"plan":[{"agent_key":"weather_assistant","sub_task":"get weather for Paris"}],'
        '"target_artifacts_source":[]}'
    )
    monkeypatch.setattr(build_plan_module, "resolve_router_provider", lambda _t: provider)

    result = await build_plan_node(
        {"thread_id": "t", "user_input": "add paris", "chat_history": [], "agent_results": []}
    )

    assert result["router_intent"] == "task"
    assert result["plan"] == [{"agent_key": "weather_assistant", "sub_task": "get weather for Paris"}]
    assert result["target_agent_key"] == "weather_assistant"
    assert result["handoff"]["target_agent_key"] == "weather_assistant"


@pytest.mark.asyncio
async def test_build_plan_drops_unknown_agent_keys_then_clarifies(monkeypatch) -> None:
    # An invented agent_key is dropped; with no valid plan and no keyword match it
    # should clarify rather than dispatch.
    provider = _PlanProvider('{"plan":[{"agent_key":"made_up_agent","sub_task":"do x"}]}')
    monkeypatch.setattr(build_plan_module, "resolve_router_provider", lambda _t: provider)

    result = await build_plan_node(
        {"thread_id": "t", "user_input": "zzqq nonsense request", "chat_history": [], "agent_results": []}
    )

    # made_up_agent is invalid → dropped. Fallback keyword heuristic may still pick a
    # default agent; if it can't, we clarify. Either way, no invalid key survives.
    plan = result.get("plan") or []
    valid = _real_agent_keys()
    assert all(step["agent_key"] in valid for step in plan)
    if not plan:
        assert result["router_intent"] == "clarify"
