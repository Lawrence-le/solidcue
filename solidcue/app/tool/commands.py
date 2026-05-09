import typer
from InquirerPy import inquirer
from rich import print
import re

from solidcue.app.utils.normalize import normalize_key
from solidcue.services.mcp_service import create_mcp_server as create_mcp_server_service, get_mcp_servers
from solidcue.services.tool_service import (
    CreateApiToolInput,
    CreateMcpToolInput,
    CreateRagToolInput,
    create_api_tool,
    create_mcp_tool,
    create_rag_tool,
    format_discovered_tool,
    get_discovered_tools_for_server,
    get_mcp_servers_for_tool_creation,
    get_tools,
    save_created_tool,
)
from solidcue.tools.schema import MCPAuthConfig, MCPServerConfig


ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _prompt_token_env_name() -> str:
    while True:
        value = typer.prompt(
            "Token environment variable name (e.g. SERPAPI_API_KEY, not the raw key)"
        ).strip()

        if not value:
            print("[red]Environment variable name cannot be empty.[/red]")
            continue

        if not ENV_VAR_NAME_PATTERN.match(value):
            print(
                "[red]Invalid env var name.[/red] Use uppercase letters, digits, and underscores only, "
                "and start with a letter or underscore."
            )
            continue

        return value


def prompt_approval_policy() -> tuple[str, str, str | None]:
    approval_mode = inquirer.select(
        message="Select approval mode:",
        choices=["never", "always", "conditional"],
        default="never",
    ).execute()

    approval_risk = inquirer.select(
        message="Select approval risk:",
        choices=["low", "medium", "high"],
        default="low",
    ).execute()

    custom_prompt = typer.prompt(
        "Approval prompt (leave blank for default)",
        default="",
    ).strip()

    return approval_mode, approval_risk, custom_prompt or None


def register(app: typer.Typer) -> None:
    app.command("create-mcp-server", rich_help_panel="Tooling")(create_mcp_server)
    app.command("list-mcp-servers", rich_help_panel="Tooling")(list_mcp_servers_cmd)
    app.command("create-tool", rich_help_panel="Tooling")(create_tool)
    app.command("list-tools", rich_help_panel="Tooling")(list_tools_cmd)


def build_auth_config() -> MCPAuthConfig:
    auth_type = inquirer.select(
        message="Select auth type:",
        choices=["none", "api_key", "bearer", "oauth"],
        default="none",
    ).execute()

    token_env = None
    location = "header"
    header_name = "Authorization"
    prefix = "Bearer"
    param_name = "api_key"

    if auth_type != "none":
        token_env = _prompt_token_env_name()

        location = inquirer.select(
            message="Where is the token sent?",
            choices=["header", "query"],
            default="header",
        ).execute()

        if location == "header":
            header_name = typer.prompt("Header name", default="Authorization")
            prefix = typer.prompt("Prefix (leave empty if none)", default="Bearer")
        elif location == "query":
            param_name = typer.prompt("Query param name", default="api_key")

    return MCPAuthConfig(
        type=auth_type,
        token_env=token_env,
        location=location,
        header_name=header_name,
        prefix=prefix,
        param_name=param_name,
    )


def create_mcp_server() -> None:
    """Create a new MCP server configuration."""
    name = typer.prompt("MCP server display name")
    server_key = normalize_key(name)

    print(f"[dim]Generated MCP server key:[/dim] {server_key}")

    url = typer.prompt("MCP server URL")
    auth = build_auth_config()

    config = MCPServerConfig(
        server_key=server_key,
        name=name,
        description="",
        url=url,
        auth=auth,
    )

    try:
        created = create_mcp_server_service(config)
    except Exception as error:
        print(f"[red]Failed to create MCP server:[/red] {error}")
        raise typer.Exit(1)

    print(f"[green]Created MCP server:[/green] {created.name}")
    print(f"[dim]Key:[/dim] {created.server_key}")
    print(f"[dim]URL:[/dim] {created.url}")
    print(f"[dim]Description:[/dim] {created.description}")
    if created.auth.type != "none" and created.auth.token_env:
        print(f"[dim]Auth env var:[/dim] {created.auth.token_env}")


def list_mcp_servers_cmd() -> None:
    """List all configured MCP servers."""
    servers = get_mcp_servers()

    if not servers:
        print("[yellow]No MCP servers found.[/yellow]")
        return

    print("[bold]Available MCP servers:[/bold]")

    for server in servers:
        print(
            f"- [cyan]{server.server_key}[/cyan] "
            f"({server.name}) "
            f"[dim]url={server.url} auth={server.auth.type}[/dim]"
        )


def create_tool() -> None:
    """Create a new tool configuration."""
    tool_type = inquirer.select(
        message="Select tool type:",
        choices=["mcp", "api", "rag"],
        default="mcp",
    ).execute()

    if tool_type == "mcp":
        config = create_mcp_tool_config()
    elif tool_type == "api":
        config = create_api_tool_config()
    elif tool_type == "rag":
        config = create_rag_tool_config()
    else:
        print(f"[red]Unsupported tool type:[/red] {tool_type}")
        raise typer.Exit(1)

    save_created_tool(config)

    print(f"[green]Created tool:[/green] {config.name}")
    print(f"[dim]Key:[/dim] {config.tool_key}")
    print(f"[dim]Type:[/dim] {config.type}")

    if config.type == "mcp" and config.mcp:
        print(f"[dim]MCP server:[/dim] {config.mcp.server_key}")
        print(f"[dim]MCP tool:[/dim] {config.mcp.tool_name}")
    elif config.type == "api" and config.api:
        print(f"[dim]API URL:[/dim] {config.api.base_url}")
        print(f"[dim]Method:[/dim] {config.api.method}")
    elif config.type == "rag":
        print("[dim]RAG tool placeholder created.[/dim]")


def create_mcp_tool_config():
    servers = get_mcp_servers_for_tool_creation()

    if not servers:
        print("[red]No MCP servers found. Create one first.[/red]")
        raise typer.Exit(1)

    server_key = inquirer.select(
        message="Select MCP server:",
        choices=[
            {
                "name": f"{server.server_key} ({server.name})",
                "value": server.server_key,
            }
            for server in servers
        ],
    ).execute()

    try:
        discovered_tools = get_discovered_tools_for_server(server_key)
    except Exception as error:
        print(f"[red]Failed to discover tools from MCP server:[/red] {error}")
        raise typer.Exit(1)

    if not discovered_tools:
        print("[red]No tools found on this MCP server.[/red]")
        raise typer.Exit(1)

    existing_tools = get_tools()
    registered_mcp_tool_names = {
        tool.mcp.tool_name
        for tool in existing_tools
        if tool.type == "mcp" and tool.mcp and tool.mcp.server_key == server_key
    }

    available_discovered_tools = [
        tool
        for tool in discovered_tools
        if tool.get("name") and tool.get("name") not in registered_mcp_tool_names
    ]

    if not available_discovered_tools:
        print("[yellow]All discovered MCP tools for this server are already registered.[/yellow]")
        raise typer.Exit(0)

    selected_tool = inquirer.select(
        message="Select MCP tool:",
        choices=[
            {
                "name": format_discovered_tool(tool),
                "value": tool,
            }
            for tool in available_discovered_tools
        ],
    ).execute()

    mcp_tool_name = selected_tool["name"]
    tool_key = normalize_key(mcp_tool_name)

    print(f"[dim]Generated tool key from MCP tool:[/dim] {tool_key}")
    print(f"[dim]Tool name from MCP:[/dim] {selected_tool.get('title') or mcp_tool_name}")
    approval_mode, approval_risk, approval_prompt = prompt_approval_policy()

    return create_mcp_tool(
        CreateMcpToolInput(
            server_key=server_key,
            selected_tool=selected_tool,
            tool_key=tool_key,
            approval_mode=approval_mode,
            approval_risk=approval_risk,
            approval_prompt=approval_prompt,
        )
    )


def create_api_tool_config():
    name = typer.prompt("Tool display name")
    tool_key = normalize_key(name)

    print(f"[dim]Generated tool key:[/dim] {tool_key}")

    description = typer.prompt("Description", default="")
    base_url = typer.prompt("API base URL")

    method = inquirer.select(
        message="Select HTTP method:",
        choices=["GET", "POST"],
        default="GET",
    ).execute()

    auth = build_auth_config()
    approval_mode, approval_risk, approval_prompt = prompt_approval_policy()

    return create_api_tool(
        CreateApiToolInput(
            name=name,
            tool_key=tool_key,
            description=description,
            base_url=base_url,
            method=method,
            auth_config=auth,
            approval_mode=approval_mode,
            approval_risk=approval_risk,
            approval_prompt=approval_prompt,
        )
    )


def create_rag_tool_config():
    print("[yellow]RAG config is not fully implemented yet.[/yellow]")
    print("[dim]Creating basic RAG ToolConfig placeholder.[/dim]")

    name = typer.prompt("Tool display name")
    tool_key = normalize_key(name)

    print(f"[dim]Generated tool key:[/dim] {tool_key}")

    description = typer.prompt("Description", default="")
    approval_mode, approval_risk, approval_prompt = prompt_approval_policy()
    return create_rag_tool(
        CreateRagToolInput(
            name=name,
            tool_key=tool_key,
            description=description,
            approval_mode=approval_mode,
            approval_risk=approval_risk,
            approval_prompt=approval_prompt,
        )
    )


def list_tools_cmd() -> None:
    """List all configured tools."""
    tools = get_tools()

    if not tools:
        print("[yellow]No tools found.[/yellow]")
        return

    print("\n[bold]Available tools[/bold]\n")

    for tool in tools:
        tool_type = tool.type.upper()

        print(f"[bold cyan][{tool_type}][/bold cyan] [bold]{tool.name}[/bold]")
        print(f"  [dim]Key:[/dim]         {tool.tool_key}")
        print(f"  [dim]Approval:[/dim]    mode={tool.approval_mode} risk={tool.approval_risk}")
        if tool.approval_prompt:
            print(f"  [dim]Approval prompt:[/dim] {tool.approval_prompt}")

        if tool.description:
            print(f"  [dim]Description:[/dim] {tool.description}")

        if tool.type == "mcp" and tool.mcp:
            print(f"  [dim]Server:[/dim]      {tool.mcp.server_key}")
            print(f"  [dim]Tool name:[/dim]   {tool.mcp.tool_name}")

        elif tool.type == "api" and tool.api:
            print(f"  [dim]URL:[/dim]         {tool.api.base_url}")
            print(f"  [dim]Method:[/dim]      {tool.api.method}")

        elif tool.type == "rag":
            print("  [dim]Config:[/dim]      RAG placeholder")

        print("")
