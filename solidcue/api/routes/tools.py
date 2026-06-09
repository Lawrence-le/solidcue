"""Tool endpoints — wraps ``solidcue.services.tool_service``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from solidcue.api.schemas import ToolEnabledRequest
from solidcue.services.tool_service import (
    CreateApiToolInput,
    CreateMcpToolInput,
    CreateRagToolInput,
    create_api_tool,
    create_mcp_tool,
    create_rag_tool,
    get_discovered_tools_for_server,
    get_mcp_servers_for_tool_creation,
    get_tools,
    save_created_tool,
)
from solidcue.tools.loader import load_tool, save_tool
from solidcue.tools.schema import MCPServerConfig, ToolConfig

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("", response_model=list[ToolConfig])
def list_all_tools() -> list[ToolConfig]:
    return get_tools()


@router.get("/mcp-servers", response_model=list[MCPServerConfig])
def list_servers_for_tool_creation() -> list[MCPServerConfig]:
    return get_mcp_servers_for_tool_creation()


@router.get("/mcp-servers/{server_key}/discovered")
def discovered_tools(server_key: str) -> list[dict[str, Any]]:
    return get_discovered_tools_for_server(server_key)


@router.post("/mcp", response_model=ToolConfig, status_code=201)
def create_mcp(input_data: CreateMcpToolInput) -> ToolConfig:
    tool = create_mcp_tool(input_data)
    save_created_tool(tool)
    return tool


@router.post("/api", response_model=ToolConfig, status_code=201)
def create_api(input_data: CreateApiToolInput) -> ToolConfig:
    tool = create_api_tool(input_data)
    save_created_tool(tool)
    return tool


@router.post("/rag", response_model=ToolConfig, status_code=201)
def create_rag(input_data: CreateRagToolInput) -> ToolConfig:
    tool = create_rag_tool(input_data)
    save_created_tool(tool)
    return tool


@router.put("/{tool_key}", response_model=ToolConfig)
def update_tool(tool_key: str, config: ToolConfig) -> ToolConfig:
    if config.tool_key != tool_key:
        raise HTTPException(
            status_code=400,
            detail=f"tool_key in body ({config.tool_key}) does not match path ({tool_key})",
        )
    try:
        load_tool(tool_key)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    save_tool(config)
    return config


@router.patch("/{tool_key}/enabled", response_model=ToolConfig)
def set_tool_enabled(tool_key: str, body: ToolEnabledRequest) -> ToolConfig:
    try:
        tool = load_tool(tool_key)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    tool.enabled = body.enabled
    save_tool(tool)
    return tool
