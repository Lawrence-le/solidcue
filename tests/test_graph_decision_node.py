from solidcue.core.graph_node import decision_node as decision_node_module
from solidcue.core.graph_node.decision_node import decision_node


def test_graph_decision_node_salvages_malformed_tool_intent(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["create_csv_file"]

    malformed_output = (
        '{"thought":"Need csv.","action":"use_tool","tool_name":"create_csv_file",'
        '"tool_input":{"content":"sku,name\\nSKU001,A","title":"Inventory",'
        '"final_answer":null,"approval_preview":null}'
    )

    def fake_run_agent(**kwargs):
        return {
            "output": malformed_output,
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "create csv",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_name"] == "create_csv_file"
    assert result["decision"]["final_answer"] is None


def test_graph_decision_node_falls_back_when_tool_not_allowed(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["search_web"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"Need tool.","action":"use_tool","tool_name":"create_csv_file",'
                '"tool_input":{"content":"x","title":"y"},"final_answer":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "create csv",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is False
    assert result["decision"]["action"] == "respond"
    assert "couldn't safely execute" in result["decision"]["final_answer"]


def test_graph_decision_node_prefixed_json_still_responds(monkeypatch) -> None:
    def fake_run_agent(**kwargs):
        return {
            "output": (
                'Response: {"thought":"No tool needed","action":"respond",'
                '"tool_name":null,"tool_input":null,"final_answer":"Done","approval_preview":null}'
            ),
            "messages": [],
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "hello",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is False
    assert result["decision"]["action"] == "respond"
    assert result["decision"]["final_answer"] == "Done"


def test_graph_decision_node_rejects_use_tool_when_required_input_missing(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["scrape_webpage"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"Need tool.","action":"use_tool","tool_stage":"context","tool_name":"scrape_webpage",'
                '"tool_input":{},"final_answer":null,"approval_preview":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "summarize this page for me",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is False
    assert result["decision"]["action"] == "respond"
    assert "required inputs were missing" in result["decision"]["final_answer"]


def test_graph_decision_node_autofills_search_query_from_user_input(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["search_web"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"Need tool.","action":"use_tool","tool_stage":"context","tool_name":"search_web",'
                '"tool_input":{},"final_answer":null,"approval_preview":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "job description linkedin 4407341275",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_name"] == "search_web"
    assert result["decision"]["tool_input"] == {"query": "job description linkedin 4407341275"}


def test_graph_decision_node_autofills_scrape_url_from_user_input(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["scrape_webpage"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"Need tool.","action":"use_tool","tool_stage":"context","tool_name":"scrape_webpage",'
                '"tool_input":{},"final_answer":null,"approval_preview":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "resume_builder",
            "user_input": "Generate a resume for https://www.linkedin.com/jobs/view/4407341275/",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_name"] == "scrape_webpage"
    assert result["decision"]["tool_input"] == {
        "url": "https://www.linkedin.com/jobs/view/4407341275/"
    }


def test_graph_decision_node_allows_artifact_generation_for_missing_content(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["create_word_document"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"Need document.","action":"use_tool","tool_stage":"artifact",'
                '"tool_name":"create_word_document","tool_input":{},"final_answer":null,'
                '"approval_preview":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "create a resume document",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_stage"] == "artifact"
    assert result["decision"]["tool_name"] == "create_word_document"


def test_graph_decision_node_overrides_mismatched_context_stage_for_artifact_tool(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["create_word_document"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"Need document.","action":"use_tool","tool_stage":"context",'
                '"tool_name":"create_word_document","tool_input":{},"final_answer":null,'
                '"approval_preview":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "create a resume document",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_stage"] == "artifact"
    assert result["decision"]["tool_name"] == "create_word_document"


def test_graph_decision_node_overrides_mismatched_artifact_stage_for_context_tool(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["search_web"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"Need search.","action":"use_tool","tool_stage":"artifact",'
                '"tool_name":"search_web","tool_input":{"query":"latest llm"},"final_answer":null,'
                '"approval_preview":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "latest llm",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_stage"] == "context"
    assert result["decision"]["tool_name"] == "search_web"


def test_graph_decision_node_forces_artifact_tool_after_validation_signal(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["search_web", "create_word_document"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"No tool needed.","action":"respond","tool_stage":null,'
                '"tool_name":null,"tool_input":null,"final_answer":"Here is a summary.",'
                '"approval_preview":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "create resume document",
            "retry_reason": (
                "ARTIFACT_REQUIRED: Request appears to require artifact output. "
                "Choose an artifact-stage tool."
            ),
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_stage"] == "artifact"
    assert result["decision"]["tool_name"] == "create_word_document"
    assert result["decision"]["tool_input"] == {}
    assert result["decision"]["final_answer"] is None
    assert all(
        not (msg.get("role") == "assistant" and isinstance(msg.get("tool_calls"), list))
        for msg in result["messages"]
    )


def test_graph_decision_node_prefers_docs_create_document_on_artifact_retry(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["drive_list_by_path", "docs_create_document", "create_word_document"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"thought":"Keep gathering.","action":"use_tool","tool_stage":"context",'
                '"tool_name":"drive_list_by_path","tool_input":{"path":"resume_agent/source/education"},'
                '"final_answer":null,"approval_preview":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "generic_assistant",
            "user_input": "Generate a tailored resume",
            "retry_reason": "ARTIFACT_REQUIRED: Must output a downloadable resume document.",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["tool_name"] == "docs_create_document"
    assert result["decision"]["tool_stage"] == "artifact"
    assert result["decision"]["tool_input"] == {}


def test_graph_decision_node_recovers_tool_calls_inside_final_answer(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["scrape_webpage"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"tool_calls":[{"tool_call_id":"call_1","tool_name":"web_scraper",'
                '"tool_args":{"url":"https://www.linkedin.com/jobs/view/4407341275/"}}],'
                '"final_answer":null}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "resume_builder",
            "user_input": "Generate a resume for https://www.linkedin.com/jobs/view/4407341275/",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_stage"] == "context"
    assert result["decision"]["tool_name"] == "scrape_webpage"
    assert result["decision"]["tool_input"] == {
        "url": "https://www.linkedin.com/jobs/view/4407341275/"
    }


def test_graph_decision_node_recovers_google_search_function_args(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["search_web"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"action":"respond","final_answer":{"tool_calls":[{"function":"google_search",'
                '"args":{"query":"job description linkedin 4407341275"}}]}}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "resume_builder",
            "user_input": "Generate a resume for https://www.linkedin.com/jobs/view/4407341275/",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_stage"] == "context"
    assert result["decision"]["tool_name"] == "search_web"
    assert result["decision"]["tool_input"] == {
        "query": "job description linkedin 4407341275"
    }


def test_graph_decision_node_recovers_bare_tool_name_from_final_answer(monkeypatch) -> None:
    class FakeAgentConfig:
        tools = ["scrape_webpage"]

    def fake_run_agent(**kwargs):
        return {
            "output": (
                '{"action":"respond","final_answer":{"tool_name":"browser_scrape"}}'
            ),
            "messages": [],
            "agent_config": FakeAgentConfig(),
        }

    monkeypatch.setattr(decision_node_module, "run_agent", fake_run_agent)

    result = decision_node(
        {
            "agent_key": "resume_builder",
            "user_input": "Generate a resume for https://www.linkedin.com/jobs/view/4407341275/",
            "attempt": 0,
        }
    )

    assert result["tool_use"] is True
    assert result["decision"]["action"] == "use_tool"
    assert result["decision"]["tool_stage"] == "context"
    assert result["decision"]["tool_name"] == "scrape_webpage"
    assert result["decision"]["tool_input"] == {
        "url": "https://www.linkedin.com/jobs/view/4407341275/"
    }
