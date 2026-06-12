from solidcue.agent_configs.loader import list_agents, load_agent


def get_agent(agent_id: str):
    return load_agent(agent_id)


def get_all_agents():
    return list_agents()
