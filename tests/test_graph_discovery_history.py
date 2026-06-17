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


def test_item_key_is_positional_so_plan_is_reusable_across_sources() -> None:
    """item_key must be a positional slot, not a source identity.

    A cached task plan references sources by item_key. For the plan to be reused
    across requests with different sources (e.g. resume for jd1 vs jd2), the same
    slot must map to the same key regardless of the underlying URL.
    """
    jd1 = _build_target_artifacts_source("generate a resume for http://jd1")
    jd2 = _build_target_artifacts_source("generate a resume for http://jd2")

    assert jd1[0]["item_key"] == "item_1"
    assert jd2[0]["item_key"] == "item_1"
    assert jd1[0]["source_ref"] != jd2[0]["source_ref"]
