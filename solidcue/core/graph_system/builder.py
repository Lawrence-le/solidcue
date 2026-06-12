from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.graph_system.nodes import final_output_node, initialize_node, intent_node
from solidcue.core.graph_system.state.schema import SystemState


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


def _route_after_intent(_state: SystemState) -> Literal["final_output"]:
    return "final_output"


def _compile_graph(checkpointer: Any) -> Any:
    graph = StateGraph(SystemState)

    graph.add_node("initialize", initialize_node)
    graph.add_node("intent", intent_node)
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("initialize")
    graph.add_conditional_edges("initialize", _route_after_initialize)
    graph.add_conditional_edges("intent", _route_after_intent)
    graph.add_edge("final_output", END)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled.with_config({"recursion_limit": _resolve_recursion_limit()})


async def build_async_system_graph() -> Any:
    checkpointer = await _get_async_checkpointer()
    return _compile_graph(checkpointer)


def build_system_graph() -> Any:
    return _compile_graph(_build_checkpointer())
