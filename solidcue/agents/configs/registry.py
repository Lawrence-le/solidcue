
from .loader import load_agent, list_agents


def get_agent(agent_id: str):
    return load_agent(agent_id)


def get_all_agents():
    return list_agents()
