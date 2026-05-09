from solidcue.core.graph_node import artifact_generation_node as artifact_generation_module
from solidcue.core.graph_node.artifact_generation_node import artifact_generation_node
from solidcue.tools.schema import MCPToolConfig, ToolConfig


class _Agent:
    provider = object()


class _Provider:
    def generate(self, messages):
        return '{"title":"Resume","content":"Generated resume content"}'


def test_artifact_generation_fills_tool_input_and_uses_persona(monkeypatch) -> None:
    captured = {}

    def fake_build_messages(**kwargs):
        captured.update(kwargs)
        return [{"role": "system", "content": kwargs["persona_text"] or ""}]

    monkeypatch.setattr(artifact_generation_module, "load_agent", lambda _: _Agent())
    monkeypatch.setattr(artifact_generation_module, "get_provider", lambda _: _Provider())
    monkeypatch.setattr(artifact_generation_module, "load_agent_persona", lambda _: "Resume persona")
    monkeypatch.setattr(artifact_generation_module, "build_artifact_generation_messages", fake_build_messages)
    monkeypatch.setattr(
        artifact_generation_module,
        "load_tool",
        lambda _: ToolConfig(
            tool_key="create_word_document",
            name="Create Word Document",
            description="Generate a Word document from text content and a title.",
            type="mcp",
            mcp=MCPToolConfig(
                server_key="file_generator",
                tool_name="create_word_document",
                input_schema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["content", "title"],
                },
            ),
        ),
    )

    result = artifact_generation_node(
        {
            "agent_key": "resume_builder",
            "user_input": "create a resume",
            "decision": {
                "action": "use_tool",
                "tool_stage": "artifact",
                "tool_name": "create_word_document",
                "tool_input": {},
                "final_answer": None,
            },
        }
    )

    assert result["decision"]["tool_input"] == {
        "title": "Resume",
        "content": "Generated resume content",
    }
    assert result["messages"][0]["tool_calls"][0]["function"]["name"] == "create_word_document"
    assert "Generated resume content" in result["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert captured["persona_text"] == "Resume persona"


def test_artifact_generation_uses_accumulated_context_evidence(monkeypatch) -> None:
    captured = {}

    def fake_build_messages(**kwargs):
        captured.update(kwargs)
        return [{"role": "system", "content": kwargs["persona_text"] or ""}]

    monkeypatch.setattr(artifact_generation_module, "load_agent", lambda _: _Agent())
    monkeypatch.setattr(artifact_generation_module, "get_provider", lambda _: _Provider())
    monkeypatch.setattr(artifact_generation_module, "load_agent_persona", lambda _: "Resume persona")
    monkeypatch.setattr(artifact_generation_module, "build_artifact_generation_messages", fake_build_messages)
    monkeypatch.setattr(
        artifact_generation_module,
        "load_tool",
        lambda _: ToolConfig(
            tool_key="create_word_document",
            name="Create Word Document",
            description="Generate a Word document from text content and a title.",
            type="mcp",
            mcp=MCPToolConfig(
                server_key="file_generator",
                tool_name="create_word_document",
                input_schema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["content", "title"],
                },
            ),
        ),
    )

    artifact_generation_node(
        {
            "agent_key": "resume_builder",
            "user_input": "create a resume",
            "context_evidence": [
                {"tool_name": "search_web", "tool_input": {"query": "jd"}, "content": "JD content"},
                {"tool_name": "google_drive", "tool_input": {"path": "x"}, "content": "Drive profile"},
            ],
            "decision": {
                "action": "use_tool",
                "tool_stage": "artifact",
                "tool_name": "create_word_document",
                "tool_input": {},
                "final_answer": None,
            },
        }
    )

    assert captured["context_evidence"] == "JD content\n\nDrive profile"


def test_artifact_generation_does_not_use_rejected_draft_as_evidence(monkeypatch) -> None:
    captured = {}

    def fake_build_messages(**kwargs):
        captured.update(kwargs)
        return [{"role": "system", "content": kwargs["persona_text"] or ""}]

    monkeypatch.setattr(artifact_generation_module, "load_agent", lambda _: _Agent())
    monkeypatch.setattr(artifact_generation_module, "get_provider", lambda _: _Provider())
    monkeypatch.setattr(artifact_generation_module, "load_agent_persona", lambda _: "Resume persona")
    monkeypatch.setattr(artifact_generation_module, "build_artifact_generation_messages", fake_build_messages)
    monkeypatch.setattr(
        artifact_generation_module,
        "load_tool",
        lambda _: ToolConfig(
            tool_key="create_word_document",
            name="Create Word Document",
            description="Generate a Word document from text content and a title.",
            type="mcp",
            mcp=MCPToolConfig(
                server_key="file_generator",
                tool_name="create_word_document",
                input_schema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["content", "title"],
                },
            ),
        ),
    )

    artifact_generation_node(
        {
            "agent_key": "resume_builder",
            "user_input": "create a resume",
            "context_evidence": [
                {"tool_name": "drive_download_file", "tool_input": {"file_id": "1"}, "content": "Source content"}
            ],
            "draft_output": "Rejected placeholder draft",
            "decision": {
                "action": "use_tool",
                "tool_stage": "artifact",
                "tool_name": "create_word_document",
                "tool_input": {},
                "final_answer": None,
            },
        }
    )

    assert captured["context_evidence"] == "Source content"


def test_artifact_generation_requires_optional_document_content(monkeypatch) -> None:
    captured = {}

    class _TitleOnlyProvider:
        def generate(self, messages):
            return '{"title":"Resume"}'

    def fake_build_messages(**kwargs):
        captured.update(kwargs)
        return [{"role": "system", "content": ""}]

    monkeypatch.setattr(artifact_generation_module, "load_agent", lambda _: _Agent())
    monkeypatch.setattr(artifact_generation_module, "get_provider", lambda _: _TitleOnlyProvider())
    monkeypatch.setattr(artifact_generation_module, "load_agent_persona", lambda _: "Resume persona")
    monkeypatch.setattr(artifact_generation_module, "build_artifact_generation_messages", fake_build_messages)
    monkeypatch.setattr(
        artifact_generation_module,
        "load_tool",
        lambda _: ToolConfig(
            tool_key="docs_create_document",
            name="Docs Create Document",
            description="Create a Google Doc and optionally seed it with text.",
            type="mcp",
            mcp=MCPToolConfig(
                server_key="google_drive",
                tool_name="docs_create_document",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["title"],
                },
            ),
        ),
    )

    result = artifact_generation_node(
        {
            "agent_key": "resume_builder",
            "user_input": "create a resume",
            "decision": {
                "action": "use_tool",
                "tool_stage": "artifact",
                "tool_name": "docs_create_document",
                "tool_input": {},
                "final_answer": None,
            },
        }
    )

    assert captured["required_fields"] == ["title", "content"]
    assert result["tool_use"] is False
    assert result["finalization_reason"] == "artifact_generation_failed"
