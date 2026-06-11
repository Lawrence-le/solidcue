from solidcue.core.graph_router.nodes.intent_router_node import intent_router_node


def test_intent_router_clarifies_capability_question_before_task() -> None:
    result = intent_router_node(
        {
            "thread_id": "thread-1",
            "user_input": "can you generate a resume for https://www.linkedin.com/jobs/view/4416496575 ?",
            "chat_history": [],
        }
    )

    assert result["router_intent"] == "clarify"
    assert result["router_next"] == "final_output"
    assert "generate the resume now" in str(result["final_response"]).casefold()


def test_intent_router_keeps_direct_instruction_as_task_when_provider_missing() -> None:
    result = intent_router_node(
        {
            "thread_id": "thread-2",
            "user_input": "generate a resume for https://www.linkedin.com/jobs/view/4416496575",
            "chat_history": [],
        }
    )

    assert result["router_intent"] == "clarify"
    assert "provider" in str(result["final_response"]).casefold()
