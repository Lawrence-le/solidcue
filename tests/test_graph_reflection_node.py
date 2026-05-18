from unittest.mock import patch

from solidcue.core.graph_node.reflection_node import reflection_node
import solidcue.core.graph_node.reflection_node as reflection_module


def test_reflection_stores_clean_evidence_for_successful_tool() -> None:
    result = reflection_node(
        {
            "phase": "source",
            "current_task": "task_1",
            "agent_key": "assistant",
            "decision": {
                "action": "use_tool",
                "tool_name": "search_web",
                "tool_input": {"query": "AI engineer jobs"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": "Job description text here",
                "error": None,
            },
        }
    )

    assert result["failure_type"] is None
    assert isinstance(result.get("context_evidence"), list)
    assert len(result["context_evidence"]) == 1
    entry = result["context_evidence"][0]
    assert entry["tool_name"] == "search_web"
    assert entry["content"] == "Job description text here"
    assert "metric_reflection" in result
    assert isinstance(result["metric_reflection"], dict)
    assert "tool_input" not in entry


def test_reflection_copies_task_evidence_role_to_evidence() -> None:
    result = reflection_node(
        {
            "phase": "source",
            "current_task": "task_1",
            "agent_key": "assistant",
            "task_plan": [
                {
                    "id": "task_1",
                    "type": "source_gathering",
                    "description": "Load candidate resume master",
                    "requires": [],
                    "evidence_role": "grounding",
                    "status": "pending",
                }
            ],
            "decision": {
                "action": "use_tool",
                "tool_name": "read_file",
                "tool_input": {"path": "resume_master.md"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": "Candidate resume master text. " * 8,
                "error": None,
            },
        }
    )

    assert result["failure_type"] is None
    assert result["context_evidence"][0]["evidence_role"] == "grounding"


def test_reflection_downgrades_file_listing_from_grounding_to_context() -> None:
    result = reflection_node(
        {
            "phase": "source",
            "current_task": "task_2",
            "agent_key": "assistant",
            "task_plan": [
                {
                    "id": "task_2",
                    "type": "source_gathering",
                    "description": "Find resume master file",
                    "requires": [],
                    "evidence_role": "grounding",
                    "status": "pending",
                }
            ],
            "decision": {
                "action": "use_tool",
                "tool_name": "drive_list_by_path",
                "tool_input": {"path": "resume_agent/source/master"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": {
                    "files": [{"id": "doc_123", "name": "resume_master"}],
                    "path": "resume_agent/source/master",
                },
                "error": None,
            },
        }
    )

    assert result["failure_type"] is None
    assert result["context_evidence"][0]["evidence_role"] == "context"


def test_reflection_keeps_downloaded_document_text_as_grounding() -> None:
    result = reflection_node(
        {
            "phase": "source",
            "current_task": "task_3",
            "agent_key": "assistant",
            "task_plan": [
                {
                    "id": "task_3",
                    "type": "source_gathering",
                    "description": "Download resume master",
                    "requires": [],
                    "evidence_role": "grounding",
                    "status": "pending",
                }
            ],
            "decision": {
                "action": "use_tool",
                "tool_name": "drive_download_file",
                "tool_input": {"file_id": "doc_123"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": {
                    "name": "resume_master",
                    "content": "Personal Information\n" + ("Darren worked on AI automation systems. " * 8),
                },
                "error": None,
            },
        }
    )

    assert result["failure_type"] is None
    assert result["context_evidence"][0]["evidence_role"] == "grounding"


def test_reflection_emits_missing_source_for_failed_tool() -> None:
    result = reflection_node(
        {
            "phase": "source",
            "agent_key": "assistant",
            "decision": {
                "action": "use_tool",
                "tool_name": "search_web",
                "tool_input": {"query": "test"},
            },
            "execution_result": {
                "success": False,
                "type": "tool_execution",
                "content": None,
                "error": "Connection error",
            },
        }
    )

    assert result["failure_type"] == "missing_source"
    assert "context_evidence" not in result


def test_reflection_emits_missing_source_for_empty_content() -> None:
    result = reflection_node(
        {
            "phase": "source",
            "agent_key": "assistant",
            "decision": {
                "action": "use_tool",
                "tool_name": "search_web",
                "tool_input": {"query": "test"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": "",
                "error": None,
            },
        }
    )

    assert result["failure_type"] == "missing_source"


def test_reflection_skips_evidence_for_artifact_phase() -> None:
    result = reflection_node(
        {
            "phase": "artifact",
            "agent_key": "assistant",
            "decision": {
                "action": "use_tool",
                "tool_name": "create_word_document",
                "tool_input": {"title": "Resume"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": {"documentId": "abc123"},
                "error": None,
            },
        }
    )

    assert result["failure_type"] is None
    assert "context_evidence" not in result


def test_reflection_cleans_web_page_noise_via_llm() -> None:
    """Web content should be cleaned; LLM call is mocked to return stripped text."""
    noisy_text = (
        "Skip to main content\n"
        "LinkedIn\n"
        "Sign in\n"
        "AI Engineer\n"
        "Sea Singapore\n"
        "Key Responsibilities\n"
        "Build AI automation solutions\n"
        "Requirements\n"
        "Python experience required\n"
        "Similar jobs\n"
        "© 2026\n"
        "Privacy Policy\n"
    )
    expected_clean = (
        "AI Engineer\n"
        "Sea Singapore\n"
        "Key Responsibilities\n"
        "Build AI automation solutions\n"
        "Requirements\n"
        "Python experience required"
    )

    with patch.object(reflection_module, "_llm_clean_text", return_value=(expected_clean, {})) as mock_clean:
        result = reflection_node(
            {
                "phase": "source",
                "current_task": "task_1",
                "agent_key": "assistant",
                "decision": {
                    "action": "use_tool",
                    "tool_name": "browser_get_text",
                    "tool_input": {"url": "https://linkedin.com/jobs/123"},
                },
                "execution_result": {
                    "success": True,
                    "type": "tool_execution",
                    "content": [{"url": "https://linkedin.com/jobs/123", "text": noisy_text}],
                    "error": None,
                },
            }
        )

    mock_clean.assert_called_once_with(noisy_text.strip(), "assistant")
    assert result["failure_type"] is None
    cleaned = result["context_evidence"][0]["content"][0]["text"]
    assert cleaned == expected_clean


def test_reflection_passes_through_non_web_content() -> None:
    """Non-web content (e.g. search results as plain string) should not trigger LLM."""
    with patch.object(reflection_module, "_llm_clean_text") as mock_clean:
        result = reflection_node(
            {
                "phase": "source",
                "current_task": "task_1",
                "agent_key": "assistant",
                "decision": {
                    "action": "use_tool",
                    "tool_name": "search_web",
                    "tool_input": {"query": "AI jobs"},
                },
                "execution_result": {
                    "success": True,
                    "type": "tool_execution",
                    "content": "Some search result text",
                    "error": None,
                },
            }
        )

    mock_clean.assert_not_called()
    assert result["failure_type"] is None
    assert result["context_evidence"][0]["content"] == "Some search result text"
    assert "tool_input" not in result["context_evidence"][0]


def test_reflection_deduplicates_context_evidence() -> None:
    """Same tool + same content should not be stored twice."""
    existing_entry = {
        "task_id": "task_1",
        "tool_name": "search_web",
        "content": "existing content",
    }

    result = reflection_node(
        {
            "phase": "source",
            "current_task": "task_1",
            "agent_key": "assistant",
            "context_evidence": [existing_entry],
            "decision": {
                "action": "use_tool",
                "tool_name": "search_web",
                "tool_input": {"query": "AI jobs"},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": "existing content",
                "error": None,
            },
        }
    )

    assert result["failure_type"] is None
    assert len(result["context_evidence"]) == 1


def test_reflection_stores_same_tool_different_content() -> None:
    """Same tool with different content should both be stored."""
    existing_entry = {
        "task_id": "task_1",
        "tool_name": "browser_get_text",
        "content": "page one content",
    }

    result = reflection_node(
        {
            "phase": "source",
            "current_task": "task_1",
            "agent_key": "assistant",
            "context_evidence": [existing_entry],
            "decision": {
                "action": "use_tool",
                "tool_name": "browser_get_text",
                "tool_input": {},
            },
            "execution_result": {
                "success": True,
                "type": "tool_execution",
                "content": "page two different content",
                "error": None,
            },
        }
    )

    assert result["failure_type"] is None
    assert len(result["context_evidence"]) == 2
