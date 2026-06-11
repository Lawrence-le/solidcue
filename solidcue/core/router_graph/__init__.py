"""Router graph package.

The router graph is the user-facing entry point. It classifies the request,
decides whether to answer directly or hand off to an agent graph, and keeps
conversation-level state separate from runtime thread state.
"""

from solidcue.core.router_graph.builder import build_async_router_graph, build_router_graph

__all__ = ["build_async_router_graph", "build_router_graph"]
