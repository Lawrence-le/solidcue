from typing import Literal

from solidcue.agents.configs.schema import AgentConfig, ProviderConfig
from solidcue.providers.base import BaseProvider
from solidcue.providers.factory import get_provider

ProviderRole = Literal["brain", "lite", "reviewer", "writer"]


def _provider_config_for_role(agent: AgentConfig, role: ProviderRole) -> ProviderConfig:
    if role == "lite" and agent.lite_provider is not None:
        return agent.lite_provider

    if role == "reviewer" and agent.reviewer_provider is not None:
        return agent.reviewer_provider

    if role == "writer" and agent.writer_provider is not None:
        return agent.writer_provider

    return agent.provider


def get_provider_for_role(agent: AgentConfig, role: ProviderRole) -> BaseProvider:
    return get_provider(_provider_config_for_role(agent, role))
