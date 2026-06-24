
from pathlib import Path
from typing import Any
import yaml

from solidcue.tools.schema import MCPServerConfig, ToolConfig

# Fields that artifact tools can generate themselves (don't need to come from state)
GENERATABLE_TOOL_FIELDS = {"content", "title", "values"}


def _resolve_input_schema(tool_config: ToolConfig) -> dict[str, Any] | None:
    # Prefer the live-refreshed schema (registry); fall back to the YAML snapshot.
    # Lazy import: schema_registry imports this module, so importing it at module
    # load time would be circular.
    try:
        from solidcue.tools.schema_registry import get_tool_input_schema

        return get_tool_input_schema(tool_config.tool_key)
    except Exception:
        return getattr(getattr(tool_config, "mcp", None), "input_schema", None)


def get_required_tool_fields(tool_config: ToolConfig) -> list[str]:
    schema = _resolve_input_schema(tool_config)
    if not isinstance(schema, dict):
        return []
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [field for field in required if isinstance(field, str) and field]


def get_missing_required_tool_fields(
    tool_config: ToolConfig,
    tool_input: dict[str, Any],
) -> list[str]:
    schema = _resolve_input_schema(tool_config)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    property_map = properties if isinstance(properties, dict) else {}

    missing: list[str] = []
    for field in get_required_tool_fields(tool_config):
        value = tool_input.get(field)
        if value is None:
            missing.append(field)
            continue
        field_schema = property_map.get(field)
        field_type = field_schema.get("type") if isinstance(field_schema, dict) else None
        if field_type == "string" and isinstance(value, str) and not value.strip():
            missing.append(field)
    return missing


def split_missing_tool_fields(fields: list[str]) -> tuple[list[str], list[str]]:
    generatable: list[str] = []
    blocking: list[str] = []
    for field in fields:
        if field in GENERATABLE_TOOL_FIELDS:
            generatable.append(field)
        else:
            blocking.append(field)
    return generatable, blocking


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
