
from typing import Any

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.prompts.decision_prompt import build_decision_messages


def run_agent(
    agent_key: str,
    user_input: str,
    retry_reason: str | None = None,
    transcript: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """
    Execution layer.

    Responsibility:
    - Load AgentConfig
    - Build messages from config + user input
    - Get provider client
    - Call LLM
    - Return raw LLM output
    """

    agent_config = load_agent(agent_key)

    messages = build_decision_messages(
        agent=agent_config,
        user_input=user_input,
        retry_reason=retry_reason,
        transcript=transcript,
        metadata=metadata,
    )

    provider = get_provider_for_role(agent_config, "decision")

    output = provider.generate(messages)

    return {
        "output": output,
        "messages": messages,
        "agent_config": agent_config,
    }
