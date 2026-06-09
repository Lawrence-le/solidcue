"""MCP server endpoints — wraps ``solidcue.services.mcp_service``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from solidcue.services.mcp_service import create_mcp_server, get_mcp_servers
from solidcue.tools.schema import MCPServerConfig

router = APIRouter(prefix="/mcp/servers", tags=["mcp"])


@router.get("", response_model=list[MCPServerConfig])
def list_servers() -> list[MCPServerConfig]:
    return get_mcp_servers()


@router.post("", response_model=MCPServerConfig, status_code=201)
def create(config: MCPServerConfig) -> MCPServerConfig:
    try:
        return create_mcp_server(config)
    except Exception as error:  # connection/discovery failure
        raise HTTPException(status_code=502, detail=str(error)) from error
