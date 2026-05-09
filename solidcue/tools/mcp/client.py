
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from solidcue.tools.schema import MCPServerConfig


class MCPClient:
    """
    MCP client for SolidCue.

    Uses the official MCP streamable HTTP client.
    """

    def __init__(self, server: MCPServerConfig):
        self.server = server

    def _tool_to_dict(self, tool: Any) -> dict[str, Any]:
        """
        Normalize MCP tool objects/dicts into a plain dictionary.

        Some MCP SDK versions expose tools as Pydantic objects.
        Others may expose dict-like objects.
        """
        def _normalize_schema(schema: Any) -> dict[str, Any] | None:
            if schema is None:
                return None
            if isinstance(schema, dict):
                return schema
            if hasattr(schema, "model_dump"):
                dumped = schema.model_dump()
                return dumped if isinstance(dumped, dict) else None
            return None

        if isinstance(tool, dict):
            return {
                "name": tool.get("name"),
                "title": tool.get("title"),
                "description": tool.get("description") or "",
                "input_schema": _normalize_schema(
                    tool.get("inputSchema") or tool.get("input_schema")
                ),
                "annotations": tool.get("annotations"),
            }

        return {
            "name": getattr(tool, "name", None),
            "title": getattr(tool, "title", None),
            "description": getattr(tool, "description", None) or "",
            "input_schema": _normalize_schema(
                getattr(tool, "inputSchema", None)
                or getattr(tool, "input_schema", None)
            ),
            "annotations": getattr(tool, "annotations", None),
        }

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        Discover tools exposed by the MCP server.

        Returns dictionaries suitable for CLI display and ToolConfig creation.
        """
        try:
            async with streamable_http_client(self.server.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to reach MCP server '{self.server.server_key}' at {self.server.url}: {exc}"
            ) from exc

        return [self._tool_to_dict(tool) for tool in result.tools]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool exposed by the MCP server.
        """
        try:
            async with streamable_http_client(self.server.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to reach MCP server '{self.server.server_key}' at {self.server.url}: {exc}"
            ) from exc

        content = []

        for item in result.content:
            if isinstance(item, dict):
                content.append(item)
            else:
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str):
                    content.append(text_value)
                else:
                    content.append(str(item))

        return {
            "tool_name": tool_name,
            "content": content,
            "structured_content": getattr(result, "structuredContent", None)
            or getattr(result, "structured_content", None),
            "is_error": getattr(result, "isError", False)
            or getattr(result, "is_error", False),
        }
