from solidcue.core.utils.source_extraction import build_target_artifacts_source as _build_target_artifacts_source


def test_build_target_artifacts_source_uses_prior_user_chat_history() -> None:
    items = _build_target_artifacts_source(
        "Archive the earlier posting.",
        chat_history=[
            {"role": "user", "content": "https://www.linkedin.com/jobs/view/4397866443"},
            {"role": "assistant", "content": "I can archive that JD."},
            {"role": "user", "content": "Please do it."},
        ],
    )

    assert len(items) == 1
    assert items[0]["source_ref"] == "https://www.linkedin.com/jobs/view/4397866443"
