from typing import Literal

from solidcue.agents.configs.schema import AgentConfig, ProviderConfig
from solidcue.providers.base import BaseProvider
from solidcue.providers.factory import get_provider

ProviderRole = Literal["decision", "sufficiency", "validator"]


def _provider_config_for_role(agent: AgentConfig, role: ProviderRole) -> ProviderConfig:
    if role == "sufficiency" and agent.sufficiency_provider is not None:
        return agent.sufficiency_provider

    if role == "validator" and agent.validator_provider is not None:
        return agent.validator_provider

    return agent.provider


def get_provider_for_role(agent: AgentConfig, role: ProviderRole) -> BaseProvider:
    return get_provider(_provider_config_for_role(agent, role))
