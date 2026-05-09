
def build_server_description(server_name: str, tools: list[dict]) -> str:
    if not tools:
        return f"{server_name} MCP server. No tools discovered."

    tool_summaries = []

    for tool in tools:
        name = tool.get("name", "unknown_tool")
        description = tool.get("description", "")

        if description:
            tool_summaries.append(f"{name}: {description}")
        else:
            tool_summaries.append(name)

    return (
        f"{server_name} MCP server exposing {len(tools)} tool(s): "
        + "; ".join(tool_summaries)
    )
