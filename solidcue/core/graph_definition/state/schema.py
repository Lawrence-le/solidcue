from __future__ import annotations

from typing import Any, Literal, TypedDict


class DefinitionState(TypedDict, total=False):
    """State for the graph_definition subgraph — one file per invocation."""

    definition_target: Literal["persona", "skill", "tools"]
    agent_key: str
    agent_spec: dict[str, Any]
    contract_skill: str
    definition_content: str
    definition_path: str
    overwrite: bool
