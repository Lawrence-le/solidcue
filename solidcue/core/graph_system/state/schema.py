from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from solidcue.core.graph_agent.state.schema import AgentState


class SystemState(AgentState, total=False):
    """State for the no-agent-key system graph.

    Inherits AgentState so it shares the request/identity/response channels with
    the agent and router graphs. This lets the router run the system graph as a
    subgraph node — the create-agent fields below round-trip through RouterState,
    which inherits SystemState.
    """

    system_intent: Literal[
        "bootstrap",
        "setup_provider",
        "create_agent",
        "import_agent",
        "select_agent",
        "repair_config",
        "final_output",
    ]
    system_next: Literal["final_output"]
    route_reason: str
    assistant_draft: str
    final_response: str
    system_skill_key: str
    system_skill_path: str
    system_skill: str

    workspace_has_agents: bool
    available_agent_keys: list[str]
    available_agents: Annotated[list[dict[str, Any]], operator.add]
    messages: Annotated[list[dict[str, Any]], operator.add]
    available_system_skill_keys: list[str]

    # create_agent branch
    agent_spec: dict[str, Any]
    artifacts: Annotated[list[dict[str, Any]], operator.add]
    created_agent_key: str
    created_config_path: str


class SystemSubgraphOutput(TypedDict, total=False):
    """Public output from the embedded system subgraph.

    The system graph still uses the shared message channels internally, but the
    parent graph should not receive them back because both channels use
    ``operator.add``. Returning them from a nested graph would append the same
    conversation entries a second time when the parent merges the subgraph
    output into its persisted state.
    """

    system_intent: Literal[
        "bootstrap",
        "setup_provider",
        "create_agent",
        "import_agent",
        "select_agent",
        "repair_config",
        "final_output",
    ]
    system_next: Literal["final_output"]
    route_reason: str
    assistant_draft: str
    final_response: str
    system_skill_key: str
    system_skill_path: str
    system_skill: str

    workspace_has_agents: bool
    available_agent_keys: list[str]
    available_agents: list[dict[str, Any]]
    available_system_skill_keys: list[str]

    agent_spec: dict[str, Any]
    artifacts: list[dict[str, Any]]
    created_agent_key: str
    created_config_path: str
