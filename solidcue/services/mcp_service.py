import asyncio

from solidcue.tools.loader import list_mcp_servers, save_mcp_server
from solidcue.tools.mcp.client import MCPClient
from solidcue.tools.schema import MCPServerConfig


def build_server_description(server_name: str, tools: list[dict]) -> str:
    if not tools:
        return f"{server_name} MCP server. No tools discovered."

    tool_names = [tool.get("name", "unknown_tool") for tool in tools]
    return (
        f"{server_name} MCP server exposing {len(tools)} tool(s): "
        f"{', '.join(tool_names)}"
    )


def discover_server_tools(config: MCPServerConfig) -> list[dict]:
    client = MCPClient(config)
    return asyncio.run(client.list_tools())


def create_mcp_server(config: MCPServerConfig) -> MCPServerConfig:
    discovered_tools = discover_server_tools(config)
    final_config = MCPServerConfig(
        server_key=config.server_key,
        name=config.name,
        description=build_server_description(config.name, discovered_tools),
        url=config.url,
        auth=config.auth,
    )
    save_mcp_server(final_config)
    return final_config


def get_mcp_servers() -> list[MCPServerConfig]:
    return list_mcp_servers()
