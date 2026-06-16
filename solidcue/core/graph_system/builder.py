from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.graph_system.nodes import (
    collect_spec_node,
    final_output_node,
    generate_definitions_node,
    initialize_node,
    intent_node,
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


def _resolve_checkpoint_db_path() -> Path:
    configured_path = os.getenv("SOLIDCUE_CHECKPOINT_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".solidcue" / "checkpoints.sqlite"


def _build_checkpointer() -> Any:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpoint_db_path = _resolve_checkpoint_db_path()
        checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(checkpoint_db_path), check_same_thread=False)
        return SqliteSaver(conn)
    except ModuleNotFoundError:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


_async_checkpointer: Any = None
_async_checkpointer_lock: asyncio.Lock | None = None


async def _get_async_checkpointer() -> Any:
    global _async_checkpointer, _async_checkpointer_lock
    if _async_checkpointer_lock is None:
        _async_checkpointer_lock = asyncio.Lock()
    async with _async_checkpointer_lock:
        if _async_checkpointer is not None:
            return _async_checkpointer
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            checkpoint_db_path = _resolve_checkpoint_db_path()
            checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(checkpoint_db_path))
            _async_checkpointer = AsyncSqliteSaver(conn)
        except ModuleNotFoundError:
            from langgraph.checkpoint.memory import InMemorySaver

            _async_checkpointer = InMemorySaver()
        return _async_checkpointer


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
    # generate_definitions writes persona/skill/tools sequentially under one span.
    graph.add_node("select_tools",         select_tools_node)
    graph.add_node("generate_definitions", generate_definitions_node)
    graph.add_node("write_config",         write_config_node)
    graph.add_node("verify",               verify_node)
    graph.add_node("final_output",         final_output_node)

    graph.set_entry_point("initialize")

    graph.add_conditional_edges("initialize", _route_after_initialize)
    graph.add_conditional_edges("intent", _route_after_intent)
    graph.add_conditional_edges("collect_spec", _route_after_collect_spec)

    graph.add_edge("select_tools", "generate_definitions")
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


async def build_async_system_graph() -> Any:
    checkpointer = await _get_async_checkpointer()
    return _compile_graph(checkpointer)


def build_system_graph() -> Any:
    return _compile_graph(_build_checkpointer())


async def build_for_server(config: Any) -> Any:
    """LangGraph Server graph factory for the system graph.

    The server injects its own checkpointer; we compile without one.
    """
    configurable = (config or {}).get("configurable") or {}
    thread_id: str | None = configurable.get("thread_id") or None
    return _compile_graph(checkpointer=None, session_id=thread_id)
