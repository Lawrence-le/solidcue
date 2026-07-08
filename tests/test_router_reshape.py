import importlib

import pytest

from solidcue.core.graph_router.builder import _route_after_intent_router

reshape_module = importlib.import_module("solidcue.core.graph_router.nodes.reshape_node")
from solidcue.core.graph_router.nodes.reshape_node import reshape_node


class _StreamProvider:
    model = "reshape-stream-model"

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def async_stream_generate(self, _messages, **_kwargs):
        for chunk in self._chunks:
            yield chunk


def _silence_stream_writer(monkeypatch) -> None:
    monkeypatch.setattr(reshape_module, "get_stream_writer", lambda: (lambda _event: None))


def test_route_sends_reshape_intent_to_reshape_node() -> None:
    assert _route_after_intent_router({"router_intent": "reshape"}) == "reshape"


def test_route_task_goes_to_build_plan() -> None:
    # Planning is decoupled: a task intent first routes to build_plan, which writes
    # the execution plan and then routes on to execute_plan.
    assert _route_after_intent_router({"router_intent": "task"}) == "build_plan"


@pytest.mark.asyncio
async def test_reshape_resynthesizes_from_retained_data(monkeypatch) -> None:
    _silence_stream_writer(monkeypatch)
    monkeypatch.setattr(
        reshape_module, "_PROFILE_ROUTER_PROVIDER", _StreamProvider(["| City | Local Time |", "\n| Tokyo | 05:15 |"])
    )

    state = {
        "user_input": "add a column for local time",
        "agent_results": [
            {
                "agent_key": "weather_assistant",
                "sub_task": "weather for Tokyo",
                "output": "Tokyo: 18.1C",
                "status": "completed",
                "data": {"successful_tool_calls": [{"content": '{"local_time": "05:15"}'}]},
            }
        ],
    }

    result = await reshape_node(state)

    assert "Local Time" in result["final_response"]
    assert result["final_response"] == result["synthesis_draft"]


@pytest.mark.asyncio
async def test_reshape_resynthesizes_from_chat_history_only(monkeypatch) -> None:
    # Data produced by a `chat` turn never lands in agent_results; it lives only in
    # CHAT_HISTORY. Reshape must still re-render it instead of dead-ending.
    _silence_stream_writer(monkeypatch)
    monkeypatch.setattr(
        reshape_module, "_PROFILE_ROUTER_PROVIDER", _StreamProvider(["| Building | Height |", "\n| Lakhta Center | 462m |"])
    )

    result = await reshape_node(
        {
            "user_input": "show me in table format",
            "agent_results": [],
            "chat_history": [
                {"role": "user", "content": "give me the top buildings in europe"},
                {"role": "assistant", "content": "Lakhta Center (462m), Federation Tower (374m)..."},
            ],
        }
    )

    assert "Building" in result["final_response"]
    assert "fetch the data again" not in result["final_response"]


@pytest.mark.asyncio
async def test_reshape_falls_back_when_no_retained_data(monkeypatch) -> None:
    _silence_stream_writer(monkeypatch)
    # Even with a provider available, missing structured data must not fabricate.
    monkeypatch.setattr(reshape_module, "_PROFILE_ROUTER_PROVIDER", _StreamProvider(["should not be used"]))

    result = await reshape_node(
        {
            "user_input": "add a column for local time",
            "agent_results": [
                {"agent_key": "weather_assistant", "output": "Tokyo: 18.1C", "status": "completed"}
            ],
        }
    )

    assert "fetch the data again" in result["final_response"]
