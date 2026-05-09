from pydantic import BaseModel
from typing import Any, cast

from solidcue.agents.configs.loader import list_agents, load_agent, save_agent, save_agent_persona
from solidcue.agents.configs.schema import AgentConfig, ProviderConfig
from solidcue.core.graph.builder import build_agent_graph
from solidcue.core.state.schema import AgentState
from solidcue.user.loader import load_user_profile
from solidcue.utils.env import generate_env_key, write_env_key
from solidcue.utils.tracing import configure_langsmith_tracing_env, trace_langgraph_invoke
from langgraph.types import Command


class CreateAgentInput(BaseModel):
    name: str
    agent_key: str
    description: str
    decision_provider_type: str
    decision_base_url: str | None
    decision_api_key: str
    decision_model: str
    sufficiency_provider_type: str
    sufficiency_base_url: str | None
    sufficiency_api_key: str
    sufficiency_model: str
    validator_provider_type: str
    validator_base_url: str | None
    validator_api_key: str
    validator_model: str
    selected_tools: list[str]


def _build_provider_config(
    *,
    provider_type: str,
    base_url: str | None,
    api_key_env: str,
    model: str,
) -> ProviderConfig:
    return ProviderConfig(
        type=provider_type,
        base_url=base_url or None,
        api_key_env=api_key_env,
        model=model,
    )


def create_agent(input_data: CreateAgentInput) -> tuple[AgentConfig, str]:
    decision_env_key = generate_env_key(f"{input_data.agent_key}_decision")
    sufficiency_env_key = generate_env_key(f"{input_data.agent_key}_sufficiency")
    validator_env_key = generate_env_key(f"{input_data.agent_key}_validator")
    write_env_key(decision_env_key, input_data.decision_api_key)
    write_env_key(sufficiency_env_key, input_data.sufficiency_api_key)
    write_env_key(validator_env_key, input_data.validator_api_key)

    decision_provider = _build_provider_config(
        provider_type=input_data.decision_provider_type,
        base_url=input_data.decision_base_url,
        api_key_env=decision_env_key,
        model=input_data.decision_model,
    )
    sufficiency_provider = _build_provider_config(
        provider_type=input_data.sufficiency_provider_type,
        base_url=input_data.sufficiency_base_url,
        api_key_env=sufficiency_env_key,
        model=input_data.sufficiency_model,
    )
    validator_provider = _build_provider_config(
        provider_type=input_data.validator_provider_type,
        base_url=input_data.validator_base_url,
        api_key_env=validator_env_key,
        model=input_data.validator_model,
    )

    config = AgentConfig(
        agent_key=input_data.agent_key,
        name=input_data.name,
        description=input_data.description,
        provider=decision_provider,
        sufficiency_provider=sufficiency_provider,
        validator_provider=validator_provider,
        tools=input_data.selected_tools,
    )
    path = save_agent(config)
    save_agent_persona(config.agent_key)
    return config, str(path)


def get_agents() -> list[AgentConfig]:
    return list_agents()


def _build_run_config(
    *,
    agent_key: str,
    profile_data: dict,
    debug: bool,
) -> dict:
    metadata = {
        "agent_key": agent_key,
        "debug": debug,
    }
    for key in ("location", "timezone"):
        value = profile_data.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value

    return {
        "run_name": f"solidcue:{agent_key}",
        "tags": ["solidcue", "langgraph", f"agent:{agent_key}"],
        "metadata": metadata,
    }


def run_agent(
    agent_key: str,
    user_input: str,
    thread_id: str,
    debug: bool = False,
) -> tuple[AgentConfig, AgentState]:
    agent = load_agent(agent_key)
    profile = load_user_profile()
    profile_data = profile.model_dump(exclude_none=True)
    state: AgentState = {
        "agent_key": agent.agent_key,
        "user_input": user_input,
        "config": profile_data,
        "max_retries": 10,
    }
    configure_langsmith_tracing_env()
    graph = build_agent_graph()
    run_config = _build_run_config(
        agent_key=agent.agent_key,
        profile_data=profile_data,
        debug=debug,
    )
    run_config["configurable"] = {"thread_id": thread_id}
    result = cast(
        AgentState,
        trace_langgraph_invoke(
            span_name="solidcue.langgraph.run_agent",
            attributes={
                "solidcue.agent_key": agent.agent_key,
                "solidcue.thread_id": thread_id,
                "solidcue.debug": debug,
            },
            invoke=lambda: graph.invoke(state, config=run_config),
        ),
    )
    return agent, result


def run_agent_step(
    *,
    agent_key: str,
    thread_id: str,
    debug: bool = False,
    user_input: str | None = None,
    resume_value: str | None = None,
) -> tuple[AgentConfig, Any]:
    """Run one LangGraph step, either initial input or resume from interrupt."""
    agent = load_agent(agent_key)
    profile = load_user_profile()
    profile_data = profile.model_dump(exclude_none=True)
    configure_langsmith_tracing_env()

    graph = build_agent_graph()
    run_config = _build_run_config(
        agent_key=agent.agent_key,
        profile_data=profile_data,
        debug=debug,
    )
    run_config["configurable"] = {"thread_id": thread_id}

    if resume_value is not None:
        return (
            agent,
            trace_langgraph_invoke(
                span_name="solidcue.langgraph.run_agent_step.resume",
                attributes={
                    "solidcue.agent_key": agent.agent_key,
                    "solidcue.thread_id": thread_id,
                    "solidcue.debug": debug,
                },
                invoke=lambda: graph.invoke(Command(resume=resume_value), config=run_config),
            ),
        )

    if user_input is None:
        raise ValueError("user_input is required for initial run")

    state: AgentState = {
        "agent_key": agent.agent_key,
        "user_input": user_input,
        "config": profile_data,
        "max_retries": 10,
    }
    return (
        agent,
        trace_langgraph_invoke(
            span_name="solidcue.langgraph.run_agent_step.initial",
            attributes={
                "solidcue.agent_key": agent.agent_key,
                "solidcue.thread_id": thread_id,
                "solidcue.debug": debug,
            },
            invoke=lambda: graph.invoke(state, config=run_config),
        ),
    )
