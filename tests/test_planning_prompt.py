from solidcue.prompts.planning_prompt import build_planning_messages


def test_planning_prompt_includes_output_paths_from_metadata() -> None:
    messages = build_planning_messages(
        user_input="Create a resume",
        metadata={
            "source_paths": ["resume_agent/source/profile"],
            "output_paths": ["resume_agent/generated_resumes/"],
            "source_filenames": ["master_resume.md"],
            "output_filenames": ["Lawrence Lee Resume.docx"],
        },
    )

    assert len(messages) == 3
    system_message = messages[0]
    runtime_message = messages[1]
    assert system_message["role"] == "system"
    content = runtime_message["content"]

    assert "Preferred Output Paths" in content
    assert "- resume_agent/generated_resumes/" in content
    assert "Preferred Source Filenames" in content
    assert "- master_resume.md" in content
    assert "Preferred Output Filenames" in content
    assert "- Lawrence Lee Resume.docx" in content
    assert "Atomic Tooling" in system_message["content"]
    assert "Synthesis Granularity" in system_message["content"]
    assert "exactly ONE synthesis task per final deliverable" in system_message["content"]
    assert "Evidence Role Rules" in system_message["content"]
    assert '"evidence_role": "grounding"' in system_message["content"]
    assert "candidate resume/profile/work history is `grounding`" in system_message["content"]
