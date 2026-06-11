from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.router_graph.router_node import (
    final_output_node,
    handoff_node,
    initialize_router_node,
    intent_router_node,
)
from solidcue.core.router_graph.state import RouterState


def _resolve_recursion_limit() -> int:
    raw = os.getenv("SOLIDCUE_RECURSION_LIMIT")
    if not raw:
        return 40
    try:
        value = int(raw)
    except ValueError:
        return 40
    return value if value > 0 else 40


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


def _route_after_initialize(_state: RouterState) -> Literal["intent_router"]:
    return "intent_router"


def _route_after_intent_router(state: RouterState) -> Literal["handoff", "final_output"]:
    if state.get("router_next") == "handoff":
        return "handoff"
    return "final_output"


def _route_after_handoff(_state: RouterState) -> Literal["final_output"]:
    return "final_output"


def _compile_graph(checkpointer: Any) -> Any:
    graph = StateGraph(RouterState)

    graph.add_node("initialize", initialize_router_node)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("handoff", handoff_node)
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("initialize")

    graph.add_conditional_edges("initialize", _route_after_initialize)
    graph.add_conditional_edges("intent_router", _route_after_intent_router)
    graph.add_conditional_edges("handoff", _route_after_handoff)
    graph.add_edge("final_output", END)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled.with_config({"recursion_limit": _resolve_recursion_limit()})


async def build_async_router_graph() -> Any:
    checkpointer = await _get_async_checkpointer()
    return _compile_graph(checkpointer)


def build_router_graph() -> Any:
    return _compile_graph(_build_checkpointer())
