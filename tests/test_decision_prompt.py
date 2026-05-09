from solidcue.prompts.decision_prompt import build_decision_messages
class DummyAgent:
    agent_key = "generic_assistant"
    name = "Generic Assistant"
    description = "Help user do basic generic tasks"
    tools = []


def test_retry_prompt_encourages_alternate_available_tool() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="when is the upcoming Arsenal match?",
        retry_reason="Error executing tool search_web: SERPAPI_API_KEY is not configured",
    )

    system_prompt = messages[0]["content"]
    assert "If another available tool can satisfy the same user request" in system_prompt
    assert "Do not repeat a failed tool call unless" in system_prompt
    assert "no untried suitable tool remains" in system_prompt
    assert "Respond with a limitation only when no available tool can help" in system_prompt
    assert "do not mention internal tool names" in system_prompt


def test_system_prompt_requires_explicit_tool_evidence() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="when is the upcoming Arsenal match?",
    )

    system_prompt = messages[0]["content"]
    assert "Treat tool outputs as evidence only for facts they explicitly contain" in system_prompt
    assert 'Freshness terms such as "current", "currently", "latest", "today", "now", or "as of" are a hard trigger' in system_prompt
    assert "Do not claim lack of live access when an available tool can check" in system_prompt
    assert "You are not the responder" in system_prompt


def test_system_prompt_excludes_agent_persona() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="hello",
    )
    system_prompt = messages[0]["content"]
    assert "Persona guidance" not in system_prompt
    assert "You are the Decision node for the Solidcue LangGraph agent" in system_prompt
    assert "Choose the next graph route" in system_prompt
    assert "tool_stage" in system_prompt


def test_system_prompt_includes_persona_source_path_hints_from_metadata() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="generate a resume",
        metadata={"persona_source_paths": ["resume_agent/source/experience.md"]},
    )
    system_prompt = messages[0]["content"]
    assert "Persona source path hints" in system_prompt
    assert "resume_agent/source/experience.md" in system_prompt
