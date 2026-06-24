from solidcue.core.graph_router.prompts.router_synthesis_prompt import (
    build_router_synthesis_messages,
)

_PRIOR = {
    "agent_key": "weather_assistant",
    "sub_task": "weather for Tokyo, London, New York",
    "status": "completed",
    "output": "table for the three cities",
    "data": {"successful_tool_calls": [{"content": {"city": "Tokyo", "temperature": "21.7C"}}]},
}
# A fresh result whose rendered text wrongly claims the other cities are missing.
_CURRENT = {
    "agent_key": "weather_assistant",
    "sub_task": "weather for Paris",
    "status": "completed",
    "output": "I need the weather data for Tokyo, London, and New York to complete this.",
    "data": {"successful_tool_calls": [{"content": {"city": "Paris", "temperature": "38.2C"}}]},
}


def test_synthesis_includes_structured_data_for_every_result() -> None:
    msgs = build_router_synthesis_messages(
        user_input="add paris", agent_results=[_PRIOR, _CURRENT], chat_history=[]
    )
    user_content = msgs[1]["content"]
    # Both results' structured values must reach the model, not just rendered text.
    assert "21.7C" in user_content
    assert "38.2C" in user_content
    assert user_content.count("data:") == 2


def test_synthesis_prompt_instructs_combine_and_ignore_false_missing() -> None:
    msgs = build_router_synthesis_messages(
        user_input="add paris", agent_results=[_PRIOR, _CURRENT], chat_history=[]
    )
    system = msgs[0]["content"]
    # Must tell the model to combine extending results and not trust a result's own
    # claim that data is missing when other results carry it.
    assert "COMBINE" in system
    assert "ignore" in system.lower()
    assert "EVERY result" in system
