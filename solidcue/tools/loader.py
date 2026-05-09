
from pathlib import Path
import yaml

from solidcue.tools.schema import MCPServerConfig, ToolConfig


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"
TOOLS_DIR = CONFIG_DIR / "tools"
MCP_SERVERS_DIR = CONFIG_DIR / "mcp_servers"


def save_mcp_server(config: MCPServerConfig) -> None:
    MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

    path = MCP_SERVERS_DIR / f"{config.server_key}.yaml"

    with path.open("w") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)


def load_mcp_server(server_key: str) -> MCPServerConfig:
    path = MCP_SERVERS_DIR / f"{server_key}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"MCP server not found: {server_key}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    return MCPServerConfig(**data)


def list_mcp_servers() -> list[MCPServerConfig]:
    MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

    servers = []

    for path in MCP_SERVERS_DIR.glob("*.yaml"):
        with path.open("r") as f:
            data = yaml.safe_load(f)

        servers.append(MCPServerConfig(**data))

    return servers


def save_tool(config: ToolConfig) -> None:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    path = TOOLS_DIR / f"{config.tool_key}.yaml"

    with path.open("w") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)


def load_tool(tool_key: str) -> ToolConfig:
    path = TOOLS_DIR / f"{tool_key}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Tool not found: {tool_key}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    return ToolConfig(**data)


def list_tools() -> list[ToolConfig]:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    tools = []

    for path in TOOLS_DIR.glob("*.yaml"):
        with path.open("r") as f:
            data = yaml.safe_load(f)

        tools.append(ToolConfig(**data))

    return tools
