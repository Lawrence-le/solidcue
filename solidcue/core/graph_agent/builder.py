import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.graph_agent.nodes.decision_node import decision_node
from solidcue.core.graph_agent.nodes.discovery_node import discovery_node
from solidcue.core.graph_agent.nodes.planning_node import planning_node
from solidcue.core.graph_agent.nodes.execution_node import execution_node
from solidcue.core.graph_agent.nodes.final_output_node import final_output_node
from solidcue.core.graph_agent.nodes.initialize_node import initialize_node
from solidcue.core.graph_agent.nodes.reflection_node import reflection_node
from solidcue.core.graph_agent.nodes.router_node import router_node
from solidcue.core.graph_agent.nodes.synthesis_node import synthesis_node
from solidcue.core.graph_agent.nodes.validation_llm_node import validation_llm_node
from solidcue.core.graph_agent.state.schema import AgentState


def _resolve_recursion_limit() -> int:
    raw = os.getenv("SOLIDCUE_RECURSION_LIMIT")
    if not raw:
        return 80
    try:
        value = int(raw)
    except ValueError:
        return 80
    return value if value > 0 else 80


def _resolve_checkpoint_db_path() -> Path:
    configured_path = os.getenv("SOLIDCUE_CHECKPOINT_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".solidcue" / "checkpoints.sqlite"


def _build_checkpointer() -> Any:
    """Sync checkpointer for the non-streaming (blocking) graph paths."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpoint_db_path = _resolve_checkpoint_db_path()
        checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(checkpoint_db_path), check_same_thread=False)
        # Set before SqliteSaver creates its tables so a fresh database is born
        # with incremental auto-vacuum (free pages reclaimable without a rewrite).
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        return SqliteSaver(conn)
    except ModuleNotFoundError:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


# Module-level async checkpointer singleton — opened once, reused across requests.
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
            # Set before AsyncSqliteSaver creates its tables so a fresh database
            # is born with incremental auto-vacuum.
            await conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            _async_checkpointer = AsyncSqliteSaver(conn)
        except ModuleNotFoundError:
            from langgraph.checkpoint.memory import InMemorySaver

            _async_checkpointer = InMemorySaver()
        return _async_checkpointer


def _route_after_decision(state: AgentState) -> Literal["execution", "router"]:
    """Route after decision based on what was planned.

    - tool call planned -> execution (run source/context tool)
    - everything else   -> router (router owns phase transitions and synthesis)
    """
    decision = state.get("decision")
    if (
        state.get("tool_use")
        and isinstance(decision, dict)
        and decision.get("action") == "use_tool"
    ):
        return "execution"

    return "router"


def _route_after_router(
    state: AgentState,
) -> Literal["decision", "synthesis", "final_output"]:
    next_node = state.get("router_next")
    if next_node in {"decision", "synthesis", "final_output"}:
        return next_node
    return "final_output"


def _passthrough_final_output_node(_state: AgentState) -> dict[str, Any]:
    return {}


def _compile_graph(
    checkpointer: Any,
    *,
    streaming_final_output: bool = False,
    session_id: str | None = None,
    include_langfuse_callbacks: bool = True,
) -> Any:
    """Build and compile the agent StateGraph with the given checkpointer."""
    graph = StateGraph(AgentState)

    graph.add_node("initialize", initialize_node)
    graph.add_node("discovery", discovery_node)
    graph.add_node("planning", planning_node)
    graph.add_node("decision", decision_node)
    graph.add_node("execution", execution_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("router", router_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("validation", validation_llm_node)
    graph.add_node(
        "final_output",
        _passthrough_final_output_node if streaming_final_output else final_output_node,
    )

    graph.set_entry_point("initialize")

    graph.add_edge("initialize", "discovery")
    graph.add_edge("discovery", "planning")
    graph.add_edge("planning", "decision")

    graph.add_conditional_edges("decision", _route_after_decision)
    graph.add_edge("execution", "reflection")
    graph.add_edge("reflection", "router")

    graph.add_conditional_edges("router", _route_after_router)

    graph.add_edge("synthesis", "validation")
    graph.add_edge("validation", "router")

    graph.add_edge("final_output", END)

    compiled = graph.compile(checkpointer=checkpointer)
    cfg: dict[str, Any] = {"recursion_limit": _resolve_recursion_limit()}
    if session_id:
        # CallbackHandler reads metadata["langfuse_session_id"] on the root chain
        # start event and groups all observations under one Langfuse session.
        cfg["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_trace_name": "solidcue:agent",
        }
    if include_langfuse_callbacks:
        from solidcue.observability.langfuse import get_langfuse_callbacks

        callbacks = get_langfuse_callbacks()
        if callbacks:
            cfg["callbacks"] = callbacks
    return compiled.with_config(cfg)


async def build_async_agent_graph(*, streaming_final_output: bool = False) -> Any:
    """Build the agent graph with the async-compatible checkpointer."""
    checkpointer = await _get_async_checkpointer()
    return _compile_graph(checkpointer, streaming_final_output=streaming_final_output)


def build_agent_graph(*, streaming_final_output: bool = False) -> Any:
    """Build the agent graph with the sync checkpointer (non-streaming paths)."""
    return _compile_graph(_build_checkpointer(), streaming_final_output=streaming_final_output)


async def build_for_server(config: Any) -> Any:
    """LangGraph Server graph factory. Called per-run with the merged run config.

    The server injects its own checkpointer; we compile without one.
    agent_key is validated against the registry here; it also flows through
    AgentState at run time so nodes can load their per-agent configuration.

    config["configurable"]["agent_key"] is populated from the assistant's saved
    config, merged with any per-run overrides.
    """
    from langchain_core.runnables import RunnableConfig  # local to avoid circular at module load

    cfg: RunnableConfig = config  # type: ignore[assignment]
    configurable = cfg.get("configurable") or {}
    agent_key = configurable.get("agent_key")
    if not agent_key:
        raise ValueError(
            "config['configurable']['agent_key'] is required. "
            "Create an assistant with config={'configurable': {'agent_key': '<key>'}}."
        )
    from solidcue.agent_configs.loader import load_agent

    load_agent(agent_key)  # raises FileNotFoundError / ValueError if not registered

    thread_id: str | None = configurable.get("thread_id") or None
    return _compile_graph(checkpointer=None, session_id=thread_id)
