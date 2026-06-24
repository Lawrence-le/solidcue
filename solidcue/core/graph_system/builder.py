from __future__ import annotations

import os
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.graph_system.nodes import (
    collect_spec_node,
    final_output_node,
    generate_definitions_node,
    initialize_node,
    intent_node,
    planning_mode_node,
    select_tools_node,
    verify_node,
    write_config_node,
)
from solidcue.core.graph_system.state.schema import SystemState, SystemSubgraphOutput


def _resolve_recursion_limit() -> int:
    raw = os.getenv("SOLIDCUE_RECURSION_LIMIT")
    if not raw:
        return 20
    try:
        value = int(raw)
    except ValueError:
        return 20
    return value if value > 0 else 20


def _route_after_initialize(_state: SystemState) -> Literal["intent"]:
    return "intent"


def _route_after_intent(state: SystemState) -> str:
    if state.get("system_intent") == "create_agent":
        return "collect_spec"
    return "final_output"


def _route_after_collect_spec(state: SystemState) -> str:
    # collect_spec sets created_agent_key on success; if missing, it set an error.
    if not state.get("created_agent_key"):
        return "final_output"
    return "select_tools"


def _assemble_graph() -> StateGraph:
    """Build the system StateGraph (nodes + edges) without compiling."""
    graph = StateGraph(SystemState, output_schema=SystemSubgraphOutput)

    graph.add_node("initialize",           initialize_node)
    graph.add_node("intent",               intent_node)
    graph.add_node("collect_spec",         collect_spec_node)
    # select_tools picks registry tools for the agent (graph_system's own LLM node);
    # planning_mode classifies static vs dynamic planning (focused LLM node);
    # generate_definitions writes persona/skill/tools sequentially under one span.
    graph.add_node("select_tools",         select_tools_node)
    graph.add_node("planning_mode",        planning_mode_node)
    graph.add_node("generate_definitions", generate_definitions_node)
    graph.add_node("write_config",         write_config_node)
    graph.add_node("verify",               verify_node)
    graph.add_node("final_output",         final_output_node)

    graph.set_entry_point("initialize")

    graph.add_conditional_edges("initialize", _route_after_initialize)
    graph.add_conditional_edges("intent", _route_after_intent)
    graph.add_conditional_edges("collect_spec", _route_after_collect_spec)

    graph.add_edge("select_tools", "planning_mode")
    graph.add_edge("planning_mode", "generate_definitions")
    graph.add_edge("generate_definitions", "write_config")
    graph.add_edge("write_config", "verify")
    graph.add_edge("verify",       "final_output")
    graph.add_edge("final_output", END)

    return graph


def _compile_graph(checkpointer: Any, *, session_id: str | None = None) -> Any:
    compiled = _assemble_graph().compile(checkpointer=checkpointer)
    cfg: dict[str, Any] = {"recursion_limit": _resolve_recursion_limit()}
    if session_id:
        cfg["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_trace_name": "solidcue:system",
        }
    from solidcue.observability.langfuse import get_langfuse_callbacks

    callbacks = get_langfuse_callbacks()
    if callbacks:
        cfg["callbacks"] = callbacks
    return compiled.with_config(cfg)


def build_system_subgraph() -> Any:
    """Compiled system graph for embedding as a node in a parent graph.

    No checkpointer (the parent owns checkpointing) and no ``with_config`` wrapper,
    so the create-agent form interrupt propagates natively up to the parent run.
    """
    return _assemble_graph().compile()


def build_system_graph() -> Any:
    """Compile the system graph with an in-memory checkpointer.

    For local/test use where no LangGraph Server is running. Production uses
    ``build_for_server`` (the server injects its own checkpointer). An in-memory
    checkpointer is enough to exercise interrupts and resume within one process.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return _compile_graph(InMemorySaver())


async def build_for_server(config: Any) -> Any:
    """LangGraph Server graph factory for the system graph.

    The server injects its own checkpointer; we compile without one.
    """
    configurable = (config or {}).get("configurable") or {}
    thread_id: str | None = configurable.get("thread_id") or None
    return _compile_graph(checkpointer=None, session_id=thread_id)
