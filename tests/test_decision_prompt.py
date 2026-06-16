from solidcue.core.graph_agent.prompts.decision_prompt import build_decision_messages
import importlib as _il; decision_prompt_module = _il.import_module("solidcue.core.graph_agent.prompts.decision_prompt")
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

    runtime_context = messages[1]["content"]
    assert "TASK STATUS: INCOMPLETE" in runtime_context
    assert "Error executing tool search_web: SERPAPI_API_KEY is not configured" in runtime_context


def test_system_prompt_requires_explicit_tool_evidence() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="when is the upcoming Arsenal match?",
    )

    system_prompt = messages[0]["content"]
    runtime_context = messages[1]["content"]
    assert "AVAILABLE TOOLS" in system_prompt
    assert "RULES" in system_prompt
    assert "Tool Use" in system_prompt
    assert "TASK GUIDANCE" in runtime_context


def test_system_prompt_excludes_agent_persona() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="hello",
    )
    system_prompt = messages[0]["content"]
    assert "Persona guidance" not in system_prompt
    assert "You are the Controller for an AI Agent" in system_prompt
    assert "OUTPUT FORMAT" in system_prompt


def test_system_prompt_includes_source_path_hints_from_metadata() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="generate a resume",
        source_paths=["resume_agent/source/experience.md"],
    )
    runtime_context = messages[1]["content"]
    assert "Source path hints" in runtime_context
    assert "resume_agent/source/experience.md" in runtime_context


def test_system_prompt_includes_output_path_and_filename_hints_from_metadata() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="generate a resume",
        output_paths=["resume_agent/generated_resumes/"],
        source_filenames=["master_resume.md"],
        output_filenames=["Lawrence Lee Resume.docx"],
    )
    runtime_context = messages[1]["content"]
    assert "Output path hints" in runtime_context
    assert "resume_agent/generated_resumes/" in runtime_context
    assert "Source filename hints" in runtime_context
    assert "master_resume.md" in runtime_context
    assert "Output filename hints" in runtime_context
    assert "Lawrence Lee Resume.docx" in runtime_context


def test_system_prompt_includes_tools_guidance_from_tools_md(monkeypatch) -> None:
    monkeypatch.setattr(
        decision_prompt_module,
        "load_agent_tools",
        lambda _agent_key: "# TOOLS.md\nUse drive_list_by_path before drive_download_file.",
    )
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="generate a resume",
    )
    system_prompt = messages[0]["content"]
    assert "Tools routing guidance" in system_prompt
    assert "drive_list_by_path before drive_download_file" in system_prompt


def test_system_prompt_includes_skill_guidance_from_skill_md(monkeypatch) -> None:
    monkeypatch.setattr(
        decision_prompt_module,
        "load_agent_skill",
        lambda _agent_key: "# SKILL.md\nUse title/filename format: YYYY-MM-DD_<name>_<company>_<role>_resume",
    )
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="generate a resume",
    )
    system_prompt = messages[0]["content"]
    assert "Skill guidance" in system_prompt
    assert "Use title/filename format" in system_prompt


def test_system_prompt_does_not_include_latest_execution_result_section() -> None:
    messages = build_decision_messages(
        agent=DummyAgent(),
        user_input="download the resume",
    )
    system_prompt = messages[0]["content"]
    assert "LATEST EXECUTION RESULT" not in system_prompt
