from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class MCPAuthConfig(BaseModel):
    """
    Defines how authentication is handled when calling an external service
    (MCP server or API tool).

    This config is used by the MCP/API client at runtime to inject credentials
    into requests (headers or query params).
    """

    type: Literal["none", "api_key", "bearer", "oauth"] = "none"
    token_env: str | None = None
    location: Literal["header", "query"] = "header"
    header_name: str = "Authorization"
    prefix: str = "Bearer"
    param_name: str = "api_key"
    oauth_provider: str | None = None
    scopes: list[str] = Field(default_factory=list)


class MCPServerConfig(BaseModel):
    """
    Represents an MCP server (connection layer).

    This defines HOW to connect to a tool provider:
    - URL
    - transport type
    - authentication

    It does NOT define which tools are used — only how to reach them.
    """

    server_key: str
    name: str
    description: str = ""
    transport: Literal["streamable_http"] = "streamable_http"
    url: str
    auth: MCPAuthConfig = Field(default_factory=MCPAuthConfig)
    enabled: bool = True


class MCPToolConfig(BaseModel):
    """
    Defines a tool that exists on an MCP server.

    This is NOT creating the tool — it references an existing tool
    exposed by the MCP server.
    """

    server_key: str
    tool_name: str
    input_schema: dict[str, Any] | None = None


class APIToolConfig(BaseModel):
    """
    Defines a direct API tool (non-MCP).

    Used when calling external HTTP APIs without MCP.
    """

    base_url: str
    method: Literal["GET", "POST"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    auth: MCPAuthConfig = Field(default_factory=MCPAuthConfig)


class ToolConfig(BaseModel):
    """
    Represents a SINGLE tool that an agent can use.

    This is the abstraction layer between:
    - Agent (decision-making)
    - Execution system (MCP/API/RAG)

    Each ToolConfig = ONE capability.

    It acts as:
    - permission layer (what agent is allowed to use)
    - abstraction layer (unifies MCP, API, RAG tools)
    """

    tool_key: str
    name: str
    description: str = ""
    type: Literal["mcp", "rag", "api"]
    enabled: bool = True
    approval_mode: Literal["never", "always", "conditional"] = "never"
    approval_risk: Literal["low", "medium", "high"] = "low"
    approval_prompt: str | None = None
    mcp: Optional[MCPToolConfig] = None
    api: Optional[APIToolConfig] = None
