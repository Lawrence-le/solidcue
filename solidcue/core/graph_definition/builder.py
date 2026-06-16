from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from solidcue.core.graph_definition.nodes.generate_node import generate_node
from solidcue.core.graph_definition.nodes.load_contract_node import load_contract_node
from solidcue.core.graph_definition.nodes.write_node import write_node
from solidcue.core.graph_definition.state.schema import DefinitionState


def build_definition_graph(definition_target: str) -> Any:
    """Build a compiled subgraph that generates and writes one agent definition file."""
    graph = StateGraph(DefinitionState)

    graph.add_node("load_contract", load_contract_node)
    graph.add_node("generate", generate_node)
    graph.add_node("write", write_node)

    graph.set_entry_point("load_contract")
    graph.add_edge("load_contract", "generate")
    graph.add_edge("generate", "write")
    graph.add_edge("write", END)

    compiled = graph.compile(checkpointer=None)
    return compiled.with_config({"configurable": {"definition_target": definition_target}})


def build_persona_graph() -> Any:
    return build_definition_graph("persona")


def build_skill_graph() -> Any:
    return build_definition_graph("skill")


def build_tools_graph() -> Any:
    return build_definition_graph("tools")
