from __future__ import annotations

import os
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.graph_router.nodes import (
    execute_plan_node,
    final_output_node,
    handoff_node,
    initialize_router_node,
    intent_router_node,
)
from solidcue.core.graph_router.state.schema import RouterState


def _resolve_recursion_limit() -> int:
    raw = os.getenv("SOLIDCUE_RECURSION_LIMIT")
    if not raw:
        return 40
    try:
        value = int(raw)
    except ValueError:
        return 40
    return value if value > 0 else 40


def _route_after_intent_router(
    state: RouterState,
) -> Literal["create_agent_system", "execute_plan", "handoff", "final_output"]:
    # create_agent intent: keep conversing (final_output) until the router has
    # gathered a ready spec (name + purpose). Once agent_spec is set, delegate to
    # the system subgraph to actually build the agent.
    if state.get("router_intent") == "create_agent":
        spec = state.get("agent_spec")
        if isinstance(spec, dict) and spec.get("agent_key") and spec.get("name"):
            return "create_agent_system"
        return "final_output"
    if state.get("router_next") == "handoff":
        # task intent with a plan → execute_plan (Wave 2 fan-out path)
        if state.get("plan"):
            return "execute_plan"
        # legacy no-plan route → old handoff stub
        return "handoff"
    return "final_output"


def _compile_graph(checkpointer: Any, *, session_id: str | None = None) -> Any:
    from solidcue.core.graph_system.builder import build_system_subgraph

    graph = StateGraph(RouterState)

    graph.add_node("initialize", initialize_router_node)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("execute_plan", execute_plan_node)
    graph.add_node("handoff", handoff_node)
    # System graph embedded as a subgraph node: runs the create_agent flow and
    # surfaces its form interrupt through this graph's run.
    graph.add_node("create_agent_system", build_system_subgraph())
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize", "intent_router")
    graph.add_conditional_edges("intent_router", _route_after_intent_router)
    graph.add_edge("execute_plan", "final_output")
    graph.add_edge("handoff", "final_output")
    graph.add_edge("create_agent_system", "final_output")
    graph.add_edge("final_output", END)

    compiled = graph.compile(checkpointer=checkpointer)
    cfg: dict[str, Any] = {"recursion_limit": _resolve_recursion_limit()}
    if session_id:
        # CallbackHandler reads metadata["langfuse_session_id"] on the root chain
        # start event and groups all observations under one Langfuse session.
        cfg["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_trace_name": "solidcue:router",
        }
    from solidcue.observability.langfuse import get_langfuse_callbacks
    callbacks = get_langfuse_callbacks()
    if callbacks:
        cfg["callbacks"] = callbacks
    return compiled.with_config(cfg)


async def build_for_server(config: Any) -> Any:
    """LangGraph Server graph factory for the router graph.

    The server injects its own checkpointer; we compile without one.
    No agent_key is needed — the router routes to all registered agents.
    """
    configurable = (config or {}).get("configurable") or {}
    thread_id: str | None = configurable.get("thread_id") or None
    return _compile_graph(checkpointer=None, session_id=thread_id)
