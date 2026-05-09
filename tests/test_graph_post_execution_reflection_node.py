from solidcue.core.graph_node import post_execution_reflection_node as reflection_module
from solidcue.core.graph_node.post_execution_reflection_node import post_execution_reflection_node


def test_reflection_forces_sufficient_for_successful_context_stage(monkeypatch) -> None:
    class _Agent:
        pass

    class _Provider:
        def generate(self, _messages):
            return (
                '{"sufficient": false, "reason": "Need full final resume content", '
                '"missing": "final_artifact"}'
            )

    monkeypatch.setattr(reflection_module, "load_agent", lambda _: _Agent())
    monkeypatch.setattr(reflection_module, "get_provider_for_role", lambda _a, _r: _Provider())

    result = post_execution_reflection_node(
        {
            "agent_key": "resume_builder",
            "user_input": "Generate a resume for https://example.com/job",
            "decision": {
                "action": "use_tool",
                "tool_stage": "context",
                "tool_name": "browser_get_text",
                "tool_input": {"url": "https://example.com/job"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": "Job description text",
                "error": None,
            },
        }
    )

    reflection = result["reflection_result"]
    assert reflection["sufficient"] is True
    assert result["finalization_reason"] == "reflection_sufficient"


def test_reflection_rejects_browser_navigation_metadata_as_source_content(monkeypatch) -> None:
    result = post_execution_reflection_node(
        {
            "agent_key": "resume_builder",
            "user_input": "Generate a resume for https://example.com/job",
            "decision": {
                "action": "use_tool",
                "tool_stage": "context",
                "tool_name": "browser_navigate",
                "tool_input": {"url": "https://example.com/job"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": {
                    "url": "https://example.com/job",
                    "title": "Example job",
                    "status": 200,
                    "ok": True,
                },
                "error": None,
            },
        }
    )

    reflection = result["reflection_result"]
    assert reflection["sufficient"] is False
    assert "metadata only" in reflection["reason"]
    assert "draft_output" not in result


def test_reflection_rejects_generic_metadata_context(monkeypatch) -> None:
    result = post_execution_reflection_node(
        {
            "agent_key": "generic_builder",
            "user_input": "Generate a document from my source files",
            "decision": {
                "action": "use_tool",
                "tool_stage": "context",
                "tool_name": "knowledge_list_files",
                "tool_input": {"path": "client/source"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": [{"id": "1", "name": "profile.md", "mimeType": "text/markdown"}],
                "error": None,
            },
        }
    )

    reflection = result["reflection_result"]
    assert reflection["sufficient"] is False
    assert "metadata only" in reflection["reason"]
