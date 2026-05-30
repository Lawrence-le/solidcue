from pydantic import BaseModel
from typing import Any, cast

from solidcue.agents.configs.loader import (
    list_agents,
    load_agent,
    save_agent,
    save_agent_persona,
    save_agent_skill,
    save_agent_tools,
)
from solidcue.agents.configs.schema import AgentConfig, ProviderConfig
from solidcue.core.graph.builder import build_agent_graph
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.debug import log_state
from solidcue.user.loader import load_user_profile
from solidcue.observability import (
    configure_langsmith_tracing_env,
    generate_env_key,
    get_langfuse_callbacks,
    trace_langgraph_invoke,
    write_env_key,
)
from langgraph.types import Command


class CreateAgentInput(BaseModel):
    name: str
    agent_key: str
    description: str
    decision_provider_type: str
    decision_base_url: str | None
    decision_api_key: str
    decision_model: str
    decision_temperature: float
    lite_provider_type: str
    lite_base_url: str | None
    lite_api_key: str
    lite_model: str
    lite_temperature: float
    reviewer_provider_type: str
    reviewer_base_url: str | None
    reviewer_api_key: str
    reviewer_model: str
    reviewer_temperature: float
    writer_provider_type: str | None = None
    writer_base_url: str | None = None
    writer_api_key: str | None = None
    writer_model: str | None = None
    writer_temperature: float | None = None
    selected_tools: list[str]


def _build_provider_config(
    *,
    provider_type: str,
    base_url: str | None,
    api_key_env: str,
    model: str,
    temperature: float | None,
) -> ProviderConfig:
    return ProviderConfig(
        type=provider_type,
        base_url=base_url or None,
        api_key_env=api_key_env,
        model=model,
        temperature=temperature,
    )


def create_agent(input_data: CreateAgentInput) -> tuple[AgentConfig, str]:
    brain_env_key = generate_env_key(f"{input_data.agent_key}_brain")
    lite_env_key = generate_env_key(f"{input_data.agent_key}_lite")
    reviewer_env_key = generate_env_key(f"{input_data.agent_key}_reviewer")
    writer_env_key = generate_env_key(f"{input_data.agent_key}_writer")
    write_env_key(brain_env_key, input_data.decision_api_key)
    write_env_key(lite_env_key, input_data.lite_api_key)
    write_env_key(reviewer_env_key, input_data.reviewer_api_key)
    if input_data.writer_api_key:
        write_env_key(writer_env_key, input_data.writer_api_key)

    brain_provider = _build_provider_config(
        provider_type=input_data.decision_provider_type,
        base_url=input_data.decision_base_url,
        api_key_env=brain_env_key,
        model=input_data.decision_model,
        temperature=input_data.decision_temperature,
    )
    lite_provider = _build_provider_config(
        provider_type=input_data.lite_provider_type,
        base_url=input_data.lite_base_url,
        api_key_env=lite_env_key,
        model=input_data.lite_model,
        temperature=input_data.lite_temperature,
    )
    reviewer_provider = _build_provider_config(
        provider_type=input_data.reviewer_provider_type,
        base_url=input_data.reviewer_base_url,
        api_key_env=reviewer_env_key,
        model=input_data.reviewer_model,
        temperature=input_data.reviewer_temperature,
    )
    writer_provider = None
    if (
        input_data.writer_provider_type
        and input_data.writer_model
        and input_data.writer_api_key
    ):
        writer_provider = _build_provider_config(
            provider_type=input_data.writer_provider_type,
            base_url=input_data.writer_base_url,
            api_key_env=writer_env_key,
            model=input_data.writer_model,
            temperature=input_data.writer_temperature,
        )

    config = AgentConfig(
        agent_key=input_data.agent_key,
        name=input_data.name,
        description=input_data.description,
        provider=brain_provider,
        lite_provider=lite_provider,
        reviewer_provider=reviewer_provider,
        writer_provider=writer_provider,
        tools=input_data.selected_tools,
    )
    path = save_agent(config)
    save_agent_persona(config.agent_key)
    save_agent_skill(config.agent_key)
    save_agent_tools(config.agent_key)
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

    run_config: dict[str, Any] = {
        "run_name": f"solidcue:{agent_key}",
        "tags": ["solidcue", "langgraph", f"agent:{agent_key}"],
        "metadata": metadata,
    }
    callbacks = get_langfuse_callbacks()
    if callbacks:
        run_config["callbacks"] = callbacks
    return run_config


def _invoke_graph(
    *,
    graph: Any,
    input_payload: Any,
    run_config: dict[str, Any],
    debug: bool,
) -> Any:
    if not debug:
        return graph.invoke(input_payload, config=run_config)

    for update in graph.stream(input_payload, config=run_config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_name, node_delta in update.items():
            if isinstance(node_delta, dict):
                log_state(str(node_name), node_delta)

    snapshot = graph.get_state(run_config)
    return snapshot.values


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
            invoke=lambda: _invoke_graph(
                graph=graph,
                input_payload=state,
                run_config=run_config,
                debug=debug,
            ),
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
                invoke=lambda: _invoke_graph(
                    graph=graph,
                    input_payload=Command(resume=resume_value),
                    run_config=run_config,
                    debug=debug,
                ),
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
            invoke=lambda: _invoke_graph(
                graph=graph,
                input_payload=state,
                run_config=run_config,
                debug=debug,
            ),
        ),
    )
