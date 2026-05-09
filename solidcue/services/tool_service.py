from pydantic import BaseModel, Field
from typing import Literal



from solidcue.tools.loader import (
    list_mcp_servers,
    list_tools,
    load_mcp_server,
    save_tool,
)
from solidcue.tools.schema import APIToolConfig, MCPToolConfig, ToolConfig, MCPAuthConfig

from solidcue.services.mcp_service import discover_server_tools


class CreateMcpToolInput(BaseModel):
    server_key: str
    selected_tool: dict
    tool_key: str
    approval_mode: Literal["never", "always", "conditional"] = "never"
    approval_risk: Literal["low", "medium", "high"] = "low"
    approval_prompt: str | None = None

class CreateApiToolInput(BaseModel):
    name: str
    tool_key: str
    description: str
    base_url: str
    method: Literal["GET", "POST"] = "GET"
    auth_config: MCPAuthConfig = Field(default_factory=MCPAuthConfig)
    approval_mode: Literal["never", "always", "conditional"] = "never"
    approval_risk: Literal["low", "medium", "high"] = "low"
    approval_prompt: str | None = None


class CreateRagToolInput(BaseModel):
    name: str
    tool_key: str
    description: str
    approval_mode: Literal["never", "always", "conditional"] = "never"
    approval_risk: Literal["low", "medium", "high"] = "low"
    approval_prompt: str | None = None


def format_discovered_tool(tool: dict) -> str:
    name = tool.get("name", "unknown_tool")
    title = tool.get("title")
    description = tool.get("description", "")

    label = f"{name}"
    if title and title != name:
        label += f" ({title})"
    if description:
        label += f" - {description}"
    return label


def get_tools() -> list[ToolConfig]:
    return list_tools()


def get_mcp_servers_for_tool_creation():
    return list_mcp_servers()


def get_discovered_tools_for_server(server_key: str) -> list[dict]:
    server = load_mcp_server(server_key)
    return discover_server_tools(server)


def create_mcp_tool(input_data: CreateMcpToolInput) -> ToolConfig:
    selected_tool = input_data.selected_tool
    mcp_tool_name = selected_tool["name"]
    tool_title = selected_tool.get("title") or mcp_tool_name
    tool_description = selected_tool.get("description", "")
    input_schema = selected_tool.get("input_schema")

    return ToolConfig(
        tool_key=input_data.tool_key,
        name=tool_title,
        description=tool_description,
        type="mcp",
        approval_mode=input_data.approval_mode,
        approval_risk=input_data.approval_risk,
        approval_prompt=input_data.approval_prompt,
        mcp=MCPToolConfig(
            server_key=input_data.server_key,
            tool_name=mcp_tool_name,
            input_schema=input_schema if isinstance(input_schema, dict) else None,
        ),
    )


def create_api_tool(input_data: CreateApiToolInput) -> ToolConfig:
    return ToolConfig(
        tool_key=input_data.tool_key,
        name=input_data.name,
        description=input_data.description,
        type="api",
        approval_mode=input_data.approval_mode,
        approval_risk=input_data.approval_risk,
        approval_prompt=input_data.approval_prompt,
        api=APIToolConfig(
            base_url=input_data.base_url,
            method=input_data.method,  # validated at CLI selector
            auth=input_data.auth_config,
        ),
    )


def create_rag_tool(input_data: CreateRagToolInput) -> ToolConfig:
    return ToolConfig(
        tool_key=input_data.tool_key,
        name=input_data.name,
        description=input_data.description,
        type="rag",
        approval_mode=input_data.approval_mode,
        approval_risk=input_data.approval_risk,
        approval_prompt=input_data.approval_prompt,
    )


def requires_tool_approval(tool: ToolConfig) -> bool:
    if tool.approval_mode == "always":
        return True
    if tool.approval_mode == "conditional":
        return tool.approval_risk in {"medium", "high"}
    return False


def save_created_tool(config: ToolConfig) -> None:
    save_tool(config)
