import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from solidcue.core.graph_node.classifier_node import classifier_node
from solidcue.core.graph_node.decision_node import decision_node
from solidcue.core.graph_node.discovery_node import discovery_node
from solidcue.core.graph_node.planning_node import planning_node
from solidcue.core.graph_node.execution_node import execution_node
from solidcue.core.graph_node.final_output_node import final_output_node
from solidcue.core.graph_node.initialize_node import initialize_node
from solidcue.core.graph_node.reflection_node import reflection_node
from solidcue.core.graph_node.router_node import router_node
from solidcue.core.graph_node.synthesis_node import synthesis_node
from solidcue.core.graph_node.validation_llm_node import validation_llm_node
from solidcue.core.state.schema import AgentState


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


def _route_after_classifier(
    state: AgentState,
) -> Literal["discovery", "planning", "final_output"]:
    if state.get("phase") == "conversational":
        next_node = state.get("router_next")
        if next_node == "planning":
            return "planning"
        return "final_output"
    return "discovery"


def _route_after_planning(
    state: AgentState,
) -> Literal["decision", "final_output"]:
    if state.get("phase") == "conversational":
        return "final_output"
    return "decision"


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


def build_agent_graph():
    """Build the SolidCue agent graph with Phase 3 task planning.

    Workflow (per AGENT_GRAPH_REDESIGN.md + Phase 3):

        initialize -> discovery -> planning -> decision
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
        synthesis -> validation -> router
        final_output -> END

        Planning node (Phase 3) decomposes user queries into task_plan with
        sequential tasks to guide multi-step workflows.
    """
    graph = StateGraph(AgentState)

    graph.add_node("initialize", initialize_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("discovery", discovery_node)
    graph.add_node("planning", planning_node)
    graph.add_node("decision", decision_node)
    graph.add_node("execution", execution_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("router", router_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("validation", validation_llm_node)
    graph.add_node("final_output", final_output_node)

    graph.set_entry_point("initialize")

    graph.add_edge("initialize", "classifier")
    graph.add_conditional_edges("classifier", _route_after_classifier)
    graph.add_edge("discovery", "planning")
    graph.add_conditional_edges("planning", _route_after_planning)

    graph.add_conditional_edges("decision", _route_after_decision)
    graph.add_edge("execution", "reflection")
    graph.add_edge("reflection", "router")

    graph.add_conditional_edges("router", _route_after_router)

    graph.add_edge("synthesis", "validation")
    graph.add_edge("validation", "router")

    graph.add_edge("final_output", END)

    compiled = graph.compile(checkpointer=_build_checkpointer())
    return compiled.with_config({"recursion_limit": _resolve_recursion_limit()})
