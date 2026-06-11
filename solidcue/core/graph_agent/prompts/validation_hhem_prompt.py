from solidcue.core.graph_agent.prompts.validation_hhem_system_prompt import build_validation_hhem_system_prompt


def build_hhem_verify_messages(failed_claims: str) -> list[dict[str, str]]:
    runtime_context = (
        "Flagged claims to evaluate:\n"
        f"{failed_claims}"
    )
    return [
        {"role": "system", "content": build_validation_hhem_system_prompt()},
        {"role": "user", "content": runtime_context},
    ]
