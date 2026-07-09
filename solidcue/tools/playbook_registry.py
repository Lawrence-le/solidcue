"""In-memory MCP playbook registry.

A tool's schema says *what* it does; a server's playbook says *how its tools
sequence* (data-dependencies, ordering, preconditions). MCP servers expose that
guidance as a readable resource with a ``playbook://`` uri. This registry fetches
those resources from each server once per process and serves them keyed by
``server_key``.

Design mirrors ``schema_registry``:
- ``refresh_all()`` hits each enabled MCP server once, reads its ``playbook://``
  resource(s), and caches the concatenated text. Per-server failures are swallowed
  so one unreachable server can't block the rest.
- ``get_server_playbook(server_key)`` / ``get_playbook_for_tool(tool_key)`` resolve
  the cached text, or ``None`` when the server exposed no playbook (or was
  unreachable at warm time).

Nothing here reads or injects into prompts — that is a separate consumer concern.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from solidcue.tools.loader import list_tools, load_mcp_server, load_tool
from solidcue.tools.mcp.client import MCPClient

logger = logging.getLogger(__name__)

_PLAYBOOK_URI_PREFIX = "playbook://"

# server_key -> concatenated playbook text (only servers that exposed one)
_PLAYBOOK_CACHE: dict[str, str] = {}
_warmed = False


def get_server_playbook(server_key: str) -> str | None:
    """Return the cached playbook text for a server, or None if it has none."""
    text = _PLAYBOOK_CACHE.get(server_key)
    return text if isinstance(text, str) and text.strip() else None


def get_playbook_for_tool(tool_key: str) -> str | None:
    """Resolve a tool to its server's playbook text, or None."""
    try:
        tool_config = load_tool(tool_key)
    except Exception:
        return None
    mcp = getattr(tool_config, "mcp", None)
    server_key = getattr(mcp, "server_key", None)
    if not server_key:
        return None
    return get_server_playbook(server_key)


async def refresh_all(*, force: bool = False) -> None:
    """Fetch playbook resources from every enabled MCP server, once per process.

    Idempotent: after the first pass ``_warmed`` is set so subsequent calls are
    no-ops unless ``force=True``. Network/read failures are logged and skipped;
    affected servers simply have no cached playbook.
    """
    global _warmed
    if _warmed and not force:
        return

    # Group enabled MCP tools by server so we hit each server once.
    by_server: dict[str, list[Any]] = defaultdict(list)
    try:
        for tool in list_tools():
            if tool.type == "mcp" and tool.enabled and tool.mcp and tool.mcp.server_key:
                by_server[tool.mcp.server_key].append(tool)
    except Exception:
        logger.exception("playbook_registry: failed to enumerate tools")
        _warmed = True
        return

    for server_key in by_server:
        try:
            server = load_mcp_server(server_key)
            if not server.enabled:
                continue
            client = MCPClient(server)
            resources = await client.list_resources()
        except Exception as exc:
            logger.warning(
                "playbook_registry: refresh skipped for server '%s': %s", server_key, exc
            )
            continue

        playbook_uris = [
            str(r.get("uri"))
            for r in resources
            if isinstance(r, dict) and str(r.get("uri", "")).startswith(_PLAYBOOK_URI_PREFIX)
        ]
        if not playbook_uris:
            continue

        parts: list[str] = []
        for uri in playbook_uris:
            try:
                text = await client.read_resource(uri)
            except Exception as exc:
                logger.warning(
                    "playbook_registry: read failed for '%s' on '%s': %s",
                    uri,
                    server_key,
                    exc,
                )
                continue
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

        if parts:
            _PLAYBOOK_CACHE[server_key] = "\n\n".join(parts)

    _warmed = True


async def ensure_playbooks_warmed() -> None:
    """Warm the cache on first use; a no-op once warmed."""
    if not _warmed:
        await refresh_all()


def reset_for_tests() -> None:
    """Clear cache + warmed flag (test helper)."""
    global _warmed
    _PLAYBOOK_CACHE.clear()
    _warmed = False
