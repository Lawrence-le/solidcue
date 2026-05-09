import typer

from solidcue.app.agent.commands import register as register_agent_commands
from solidcue.app.setup.commands import register as register_setup_commands
from solidcue.app.tool.commands import register as register_tool_commands



app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """SolidCue CLI.

    Manage agents, MCP servers, tools, and run agent workflows.
    """
    pass


register_agent_commands(app)
register_tool_commands(app)
register_setup_commands(app)
