"""Run the create-agent build as a top-level router node.

Spec gathering (with its interrupt) runs in the embedded collect subgraph; the
*build* runs here, at the router level, so the build nodes' progress events reach
the client. Custom events from a nested subgraph are filtered out (the client
streams with subgraphs off), but this node runs top-level — exactly like
execute_plan — so its get_stream_writer() events surface in the existing panel.

It drives the same graph_system build nodes in sequence, threading state, and
returns the final result for final_output.
"""

from __future__ import annotations

from typing import Any

from solidcue.core.graph_router.state.schema import RouterState


async def create_agent_build_node(state: RouterState) -> dict[str, Any]:
    from solidcue.core.graph_system.nodes.generate_definition_nodes import (
        generate_definitions_node,
    )
    from solidcue.core.graph_system.nodes.planning_mode_node import planning_mode_node
    from solidcue.core.graph_system.nodes.select_tools_node import select_tools_node
    from solidcue.core.graph_system.nodes.verify_node import verify_node
    from solidcue.core.graph_system.nodes.write_config_node import write_config_node

    working: dict[str, Any] = dict(state)

    # select_tools emits the build plan + step 0; planning_mode has no step.
    working.update(await select_tools_node(working))
    working.update(await planning_mode_node(working))
    working.update(await generate_definitions_node(working))  # step 1

    working.update(write_config_node(working))  # step 2
    if not working.get("created_config_path"):
        # write_config failed — surface its error, skip verify (no half-agent report).
        return {
            "final_response": working.get("final_response"),
            "assistant_draft": working.get("assistant_draft"),
        }

    working.update(verify_node(working))  # step 3

    return {
        "created_agent_key": working.get("created_agent_key"),
        "created_config_path": working.get("created_config_path"),
        "artifacts": working.get("artifacts") or [],
        "agent_spec": working.get("agent_spec"),
        "final_response": working.get("final_response"),
        "assistant_draft": working.get("assistant_draft"),
    }
