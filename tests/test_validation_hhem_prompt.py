from solidcue.prompts.validation_hhem_prompt import build_hhem_verify_messages


def test_hhem_verify_prompt_defines_false_positives_and_real_failures() -> None:
    messages = build_hhem_verify_messages(
        failed_claims='- "Technical Skills" (score: 0.10)',
    )

    system_prompt = messages[0]["content"]
    runtime_prompt = messages[1]["content"]
    assert "REAL FAILURE" in system_prompt
    assert "FALSE POSITIVE" in system_prompt
    assert "TRUE POSITIVES" in system_prompt
    assert "Do NOT return false positives in `real_failures`" in system_prompt
    assert "If all flagged items are FALSE POSITIVES" in system_prompt
    assert '"real_failures": []' in system_prompt
    assert "Your job is NOT to re-check the source context" in system_prompt
    assert 'Technical Skills' in runtime_prompt
    assert "Candidate used Python" not in runtime_prompt
