"""In-memory MCP tool-schema registry.

The tool YAML configs carry a hand-authored `input_schema` snapshot. That snapshot
can drift from what the live MCP server actually exposes. This registry refreshes
those schemas from the servers once per process (lazily, on first use), and serves
as the single resolution point for "what is the input_schema for this tool?".

Design:
- `ensure_schemas_warmed()` fetches each MCP server's live tool list ONCE per
  process and caches `input_schema` keyed by tool_key. Per-server failures are
  swallowed so one unreachable server can't block the rest.
- `get_tool_input_schema(tool_key)` returns the refreshed schema if present, else
  falls back to the YAML snapshot. So callers always get *something* — fresh when
  the server was reachable at warm time, the YAML blueprint otherwise.

The YAML files are never mutated; the cache lives only in memory.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from solidcue.tools.loader import list_tools, load_mcp_server, load_tool
from solidcue.tools.mcp.client import MCPClient

logger = logging.getLogger(__name__)

# tool_key -> input_schema (only entries successfully fetched from a live server)
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}
_warmed = False


def _yaml_input_schema(tool_key: str) -> dict[str, Any] | None:
    try:
        tool_config = load_tool(tool_key)
    except Exception:
        return None
    schema = getattr(getattr(tool_config, "mcp", None), "input_schema", None)
    return schema if isinstance(schema, dict) else None


def get_tool_input_schema(tool_key: str) -> dict[str, Any] | None:
    """Resolve a tool's input_schema: live-refreshed if available, else YAML."""
    cached = _SCHEMA_CACHE.get(tool_key)
    if isinstance(cached, dict):
        return cached
    return _yaml_input_schema(tool_key)


async def refresh_all(*, force: bool = False) -> None:
    """Fetch live tool schemas from every enabled MCP server, once per process.

    Idempotent: after the first successful pass `_warmed` is set so subsequent
    calls are no-ops unless `force=True`. Network failures are logged and skipped;
    affected tools simply keep falling back to their YAML snapshot.
    """
    global _warmed
    if _warmed and not force:
        return

    # Group enabled MCP tools by their server so we hit each server once.
    by_server: dict[str, list[Any]] = defaultdict(list)
    try:
        for tool in list_tools():
            if tool.type == "mcp" and tool.enabled and tool.mcp and tool.mcp.server_key:
                by_server[tool.mcp.server_key].append(tool)
    except Exception:
        logger.exception("schema_registry: failed to enumerate tools")
        _warmed = True
        return

    for server_key, tools in by_server.items():
        try:
            server = load_mcp_server(server_key)
            if not server.enabled:
                continue
            live = await MCPClient(server).list_tools()
        except Exception as exc:
            # Server unreachable / disabled — those tools fall back to YAML.
            logger.warning("schema_registry: refresh skipped for server '%s': %s", server_key, exc)
            continue

        live_by_name = {
            d.get("name"): d.get("input_schema")
            for d in live
            if isinstance(d, dict) and d.get("name")
        }
        for tool in tools:
            schema = live_by_name.get(tool.mcp.tool_name)
            if isinstance(schema, dict):
                _SCHEMA_CACHE[tool.tool_key] = schema

    _warmed = True


async def ensure_schemas_warmed() -> None:
    """Warm the cache on first use; a no-op once warmed."""
    if not _warmed:
        await refresh_all()


def reset_for_tests() -> None:
    """Clear cache + warmed flag (test helper)."""
    global _warmed
    _SCHEMA_CACHE.clear()
    _warmed = False
