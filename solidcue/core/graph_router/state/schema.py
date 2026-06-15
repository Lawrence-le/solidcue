from __future__ import annotations

from typing import Any, Literal, TypedDict, Annotated
import operator

from solidcue.core.graph_agent.state.schema import AgentState


class RouterState(AgentState, total=False):
    """State for the user-facing router graph."""

    # Router intent:
    # - chat: answer directly as a normal conversation
    # - task: hand off to an agent graph to complete work
    # - create_agent: collect or route toward agent creation
    # - clarify: ask the user for missing details before routing
    # Conceptually these intents split into:
    # - before_task: new request, guidance, or setup
    # - after_task: follow-up, refinement, or resume on prior work
    router_intent: Literal["chat", "task", "create_agent", "clarify"]
    router_next: Literal["handoff", "final_output"]
    target_agent_key: str
    route_reason: str
    handoff: dict[str, Any]
    assistant_draft: str
    messages: Annotated[list[dict[str, Any]], operator.add]

    # Orchestration: the router acts as a manager. `plan` lists the sub-agents
    # (agent graphs) to dispatch for a task, each with the sub-task it should do.
    # `agent_results` is append-only — one entry per worker once it finishes.
    # `synthesis_draft` is the router's final user-facing answer composed from
    # all worker outputs.
    plan: list[dict[str, Any]]
    agent_results: Annotated[list[dict[str, Any]], operator.add]
    synthesis_draft: str
