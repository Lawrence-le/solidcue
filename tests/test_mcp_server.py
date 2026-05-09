import asyncio

from solidcue.tools.loader import load_tool, load_mcp_server
from solidcue.tools.mcp.client import MCPClient


async def main():
    tool = load_tool("create_pdf_document")
    server = load_mcp_server(tool.mcp.server_key)

    client = MCPClient(server)

    result = await client.call_tool(
        tool.mcp.tool_name,
        {
            "content": "Hello world",
            "title": "Test PDF",
        },
    )

    print("\n=== TOOL RESULT ===")
    print("Tool:", result["tool_name"])
    print("Error:", result["is_error"])

    print("\nContent:")
    for c in result["content"]:
        print("-", c)

    print("\nStructured:")
    print(result["structured_content"])


asyncio.run(main())
