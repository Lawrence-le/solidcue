import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.graph_node.decision_node import decision_node
from solidcue.core.graph_node.discovery_node import discovery_node
from solidcue.core.graph_node.artifact_generation_node import artifact_generation_node
from solidcue.core.graph_node.artifact_execution_node import artifact_execution_node
from solidcue.core.graph_node.execution_node import execution_node
from solidcue.core.graph_node.final_output_node import final_output_node
from solidcue.core.graph_node.initialize_node import initialize_node
from solidcue.core.graph_node.post_execution_reflection_node import post_execution_reflection_node
from solidcue.core.graph_node.router_node import router_node
from solidcue.core.graph_node.synthesis_node import synthesis_node
from solidcue.core.graph_node.validation_node import validation_node
from solidcue.core.state.schema import AgentState


def _resolve_checkpoint_db_path() -> Path:
    configured_path = os.getenv("SOLIDCUE_CHECKPOINT_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".solidcue" / "checkpoints.sqlite"


def _build_checkpointer() -> Any:
    """Create a persistent sqlite checkpointer when available.

    Falls back to in-memory checkpointing if sqlite extras are not installed.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpoint_db_path = _resolve_checkpoint_db_path()
        checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(checkpoint_db_path), check_same_thread=False)
        return SqliteSaver(conn)
    except ModuleNotFoundError:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


def _route_after_decision(state: AgentState) -> Literal["execution", "router", "synthesis"]:
    """Route after decision based on what was planned.

    - artifact_plan present  -> router (move to artifact phase)
    - tool call planned      -> execution (run source/context tool)
    - nothing to run         -> synthesis (respond directly)
    """
    if state.get("artifact_plan"):
        return "router"

    decision = state.get("decision")
    if (
        state.get("tool_use")
        and isinstance(decision, dict)
        and decision.get("action") == "use_tool"
    ):
        return "execution"

    return "synthesis"


def _route_after_router(
    state: AgentState,
) -> Literal["decision", "artifact_generation", "artifact_execution", "synthesis", "final_output"]:
    next_node = state.get("router_next")
    if next_node in {
        "decision",
        "artifact_generation",
        "artifact_execution",
        "synthesis",
        "final_output",
    }:
        return next_node
    return "final_output"


def build_agent_graph():
    """Build the SolidCue agent graph.

    Target workflow (per AGENT_GRAPH_REDESIGN.md):

        initialize -> discovery -> decision
                                     |
                                     +--> execution -> reflection -> router
                                     +--> synthesis -> validation -> router
                                     +--> router (when artifact_plan ready)

        router dispatches to:
            - decision           (need more source)
            - artifact_generation (artifact phase)
            - artifact_execution  (artifact retry)
            - synthesis           (synthesis phase)
            - final_output        (terminal)

        artifact_generation -> artifact_execution -> validation -> router
        final_output -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("initialize", initialize_node)
    graph.add_node("discovery", discovery_node)
    graph.add_node("decision", decision_node)
    graph.add_node("execution", execution_node)
    graph.add_node("reflection", post_execution_reflection_node)
    graph.add_node("router", router_node)
    graph.add_node("artifact_generation", artifact_generation_node)
    graph.add_node("artifact_execution", artifact_execution_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("validation", validation_node)
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize", "discovery")
    graph.add_edge("discovery", "decision")

    graph.add_conditional_edges("decision", _route_after_decision)
    graph.add_edge("execution", "reflection")
    graph.add_edge("reflection", "router")

    graph.add_conditional_edges("router", _route_after_router)

    graph.add_edge("artifact_generation", "artifact_execution")
    graph.add_edge("artifact_execution", "validation")
    graph.add_edge("synthesis", "validation")
    graph.add_edge("validation", "router")

    graph.add_edge("final_output", END)

    return graph.compile(checkpointer=_build_checkpointer())
