from solidcue.prompts.classifier_prompt import build_classifier_messages
from solidcue.prompts.classifier_system_prompt import build_classifier_system_prompt


def test_classifier_prompt_filters_assistant_history_and_current_turn() -> None:
    messages = build_classifier_messages(
        user_input="generate a resume for https://www.linkedin.com/jobs/view/4397670577",
        chat_history=[
            {"role": "user", "content": "what can you do"},
            {"role": "assistant", "content": "I can build resumes and archive JDs."},
            {"role": "user", "content": "generate a resume for https://www.linkedin.com/jobs/view/4397670577"},
        ],
    )

    runtime_context = messages[1]["content"]
    assert "what can you do" in runtime_context
    assert "I can build resumes and archive JDs." not in runtime_context
    assert runtime_context.count("generate a resume for https://www.linkedin.com/jobs/view/4397670577") == 0


def test_classifier_system_prompt_prioritizes_latest_task_turn() -> None:
    prompt = build_classifier_system_prompt()

    assert "LATEST user message" in prompt
    assert "If the latest message includes a URL" in prompt
    assert "follow-up request to perform work is **task**" in prompt
