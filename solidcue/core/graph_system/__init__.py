"""System graph package.

This graph owns no-agent-key workflows such as bootstrap, setup, recovery,
and entry-point guidance before a finalized agent exists.
"""

from solidcue.core.graph_system.builder import build_for_server, build_system_graph
from solidcue.core.graph_system.state.schema import SystemState

__all__ = ["build_for_server", "build_system_graph", "SystemState"]
