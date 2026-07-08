from __future__ import annotations

import os
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.graph_router.nodes import (
    build_plan_node,
    create_agent_build_node,
    execute_plan_node,
    final_output_node,
    handoff_node,
    initialize_router_node,
    intent_router_node,
    reshape_node,
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
) -> Literal["create_agent_collect", "build_plan", "reshape", "final_output"]:
    # Routing is purely on the classified intent — the intent router no longer builds
    # the plan; that is build_plan_node's job (reached only for the task intent).
    intent = state.get("router_intent")
    if intent == "create_agent":
        # Keep conversing (final_output) until a ready spec (name + purpose) exists;
        # once agent_spec is set, run the collect subgraph (gathers/validates + may
        # interrupt), then the top-level build node.
        spec = state.get("agent_spec")
        if isinstance(spec, dict) and spec.get("agent_key") and spec.get("name"):
            return "create_agent_collect"
        return "final_output"
    if intent == "reshape":
        # Re-present retained data without re-dispatching, from agent_results[].data.
        return "reshape"
    if intent == "task":
        # Write the execution plan in a dedicated node, then execute it.
        return "build_plan"
    return "final_output"


def _route_after_collect(state: RouterState) -> Literal["create_agent_build", "final_output"]:
    # collect_spec sets created_agent_key on a valid spec; otherwise it interrupted
    # (already surfaced) or set an error. Only proceed to the build when ready.
    if state.get("created_agent_key"):
        return "create_agent_build"
    return "final_output"


def _route_after_build_plan(
    state: RouterState,
) -> Literal["execute_plan", "final_output"]:
    # build_plan produced a plan -> execute it; otherwise it set a clarify message.
    if state.get("plan"):
        return "execute_plan"
    return "final_output"


def _compile_graph(checkpointer: Any, *, session_id: str | None = None) -> Any:
    from solidcue.core.graph_system.builder import build_collect_subgraph

    graph = StateGraph(RouterState)

    graph.add_node("initialize", initialize_router_node)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("build_plan", build_plan_node)
    graph.add_node("execute_plan", execute_plan_node)
    graph.add_node("reshape", reshape_node)
    graph.add_node("handoff", handoff_node)
    # create_agent runs in two parts: collect (embedded subgraph, may interrupt for
    # the spec form) then build (top-level node, so its progress events stream).
    graph.add_node("create_agent_collect", build_collect_subgraph())
    graph.add_node("create_agent_build", create_agent_build_node)
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize", "intent_router")
    graph.add_conditional_edges("intent_router", _route_after_intent_router)
    graph.add_conditional_edges("build_plan", _route_after_build_plan)
    graph.add_conditional_edges("create_agent_collect", _route_after_collect)
    graph.add_edge("execute_plan", "final_output")
    graph.add_edge("reshape", "final_output")
    graph.add_edge("handoff", "final_output")
    graph.add_edge("create_agent_build", "final_output")
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
