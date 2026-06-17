import asyncio
import json
from typing import Any

import typer
from InquirerPy import inquirer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from solidcue.agent_configs.loader import get_persona_path, get_skill_path, get_tools_path
from solidcue.app.utils.helpers import print_select_hint
from solidcue.app.utils.normalize import normalize_key
from solidcue.providers.config import PROVIDER_META
from solidcue.services.agent_service import CreateAgentInput, create_agent as create_agent_service
from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    build_state_snapshot,
    get_latest_thread_id,
    list_agent_state_keys,
)
from solidcue.services.workspace_service import get_agents
from solidcue.tools.loader import list_tools

console = Console(
    theme=Theme(
        {
            "markdown.text": "white",
            "markdown.paragraph": "white",
            "markdown.item": "white",
            "markdown.item.bullet": "bright_white",
            "markdown.h1": "bold bright_white",
            "markdown.h2": "bold bright_white",
            "markdown.h3": "bold bright_white",
            "markdown.strong": "bold bright_white",
            "markdown.link": "bold bright_cyan underline",
            "markdown.code": "bright_white",
            "markdown.code_block": "bright_white",
            "markdown.block_quote": "white",
        }
    )
)


def _print_section(title: str) -> None:
    print(f"\n[bold cyan]{title}[/bold cyan]")


def _print_value(label: str, value: Any) -> None:
    print(f"[bold bright_white]{label}[/bold bright_white]:")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    else:
        print(value)


def register(app: typer.Typer) -> None:
    app.command("create-agent", rich_help_panel="Agent")(create_agent)
    app.command("list-agents", rich_help_panel="Agent")(list_agents_cmd)
    app.command("run-agent", rich_help_panel="Agent")(run_agent_cmd)
    app.command("snap", rich_help_panel="Debug")(snap_cmd)


def run_agent_cmd(
    input_text: str | None = typer.Argument(None, help="Message to send to the router graph"),
    server: str = typer.Option("http://localhost:2024", "--server", help="LangGraph Server URL"),
    debug: bool = typer.Option(False, "--debug", help="Show thread, run, node, and metric debug details."),
) -> None:
    """Stream a run through the LangGraph Server router graph and print the response."""
    if not input_text:
        input_text = inquirer.text(message="Message to send:").execute()
    if not input_text or not input_text.strip():
        print("[red]Missing message input.[/red]")
        raise typer.Exit(1)
    asyncio.run(_run_agent_async(input_text.strip(), server, debug=debug))


def _chunk_field(chunk: Any, field: str) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(field)
    return getattr(chunk, field, None)


def _print_metric_summary(values: dict[str, Any]) -> None:
    events = values.get("metric_usage_events")
    if not isinstance(events, list) or not events:
        print("[dim]No metric usage events found in final thread state.[/dim]")
        return

    table = Table(title="Metric usage")
    table.add_column("Node")
    table.add_column("Model")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Estimated", justify="right")
    table.add_column("Time", justify="right")

    for event in events:
        if not isinstance(event, dict):
            continue
        table.add_row(
            str(event.get("node") or ""),
            str(event.get("model") or ""),
            str(event.get("prompt_tokens") or 0),
            str(event.get("completion_tokens") or 0),
            str(event.get("total_tokens") or 0),
            str(event.get("estimated_total") or 0),
            f"{float(event.get('time_s') or 0):.2f}s",
        )
    console.print(table)


async def _run_agent_async(user_input: str, server_url: str, *, debug: bool = False) -> None:
    try:
        from langgraph_sdk import get_client as lg_get_client
    except ImportError:
        print("[red]langgraph-sdk not installed. Run: uv add langgraph-sdk[/red]")
        raise typer.Exit(1)

    client = lg_get_client(url=server_url)

    try:
        results = await client.assistants.search(graph_id="router", limit=1)
    except Exception as exc:
        print(f"[red]LangGraph Server not reachable at {server_url}: {exc}[/red]")
        raise typer.Exit(1)

    if results:
        assistant_id = results[0]["assistant_id"]
    else:
        a = await client.assistants.create(graph_id="router", config={})
        assistant_id = a["assistant_id"]

    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    print(f"[dim]thread: {thread_id}[/dim]")
    if debug:
        print(f"[dim]server: {server_url}[/dim]")
        print(f"[dim]assistant: {assistant_id}[/dim]")
    print(f"[dim]> {user_input}[/dim]\n")

    final_response: str | None = None
    run_id: str | None = None

    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"user_input": user_input},
        stream_mode=["updates", "messages", "custom"],
        stream_resumable=True,
    ):
        event = _chunk_field(chunk, "event")
        data = _chunk_field(chunk, "data")

        if event == "metadata" and isinstance(data, dict):
            run_id = data.get("run_id") or run_id
            if debug and run_id:
                print(f"[dim]run: {run_id}[/dim]")

        elif event == "updates" and isinstance(data, dict):
            if debug:
                for node in data:
                    print(f"[dim]node: {node}[/dim]")
            if "final_output" in data:
                final_response = (data["final_output"] or {}).get("final_response") or final_response

        elif event in ("messages", "messages/partial") and isinstance(data, list) and data:
            item = data[0]
            msg = item[0] if isinstance(item, list) else item
            content = (msg or {}).get("content", "") if isinstance(msg, dict) else ""
            if isinstance(content, str) and content:
                console.print(content, end="", highlight=False)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        console.print(block.get("text", ""), end="", highlight=False)

    print("")
    if final_response:
        print("\n[bold bright_white]Response:[/bold bright_white]")
        from rich.markdown import Markdown
        console.print(Markdown(final_response))

    if debug:
        try:
            snapshot = await client.threads.get_state(thread_id)
            values = snapshot.get("values") if isinstance(snapshot, dict) else getattr(snapshot, "values", None)
            if isinstance(values, dict):
                _print_metric_summary(values)
        except Exception as exc:
            print(f"[yellow]Could not load final debug state: {exc}[/yellow]")


def snap_cmd(
    decision: bool = typer.Option(False, "--decision", help="Shortcut for decision-related state keys."),
    key: list[str] | None = typer.Option(None, "--key", help="State key to include (repeatable)."),
    all_keys: bool = typer.Option(False, "--all", help="Include all AgentState schema keys."),
    list_keys: bool = typer.Option(False, "--list-keys", help="List available AgentState keys and exit."),
    live: bool = typer.Option(False, "--live", help="Read live state from LangGraph checkpoint for a thread."),
    thread_id: str | None = typer.Option(None, "--thread-id", help="Thread id to read when using --live."),
    latest_thread: bool = typer.Option(False, "--latest-thread", help="Use the most recent checkpointed thread id."),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON output."),
) -> None:
    """Print example state snapshots for quick debugging."""
    if list_keys:
        keys = list_agent_state_keys()
        if as_json:
            print(json.dumps({"available_keys": keys}, indent=2, ensure_ascii=False))
            return
        _print_section("AVAILABLE STATE KEYS")
        for item in keys:
            print(f"- {item}")
        return

    selected_keys = list(key or [])
    if decision:
        selected_keys.extend(
            [
                "phase",
                "current_task",
                "router_next",
                "retry_reason",
                "task_plan",
                "metadata",
                "tool_call_history",
                "decision",
                "active_tool_call",
                "execution_result",
            ]
        )
    if live:
        resolved_thread_id = thread_id.strip() if isinstance(thread_id, str) and thread_id.strip() else None
        if resolved_thread_id is None and latest_thread:
            resolved_thread_id = asyncio.run(get_latest_thread_id())
        if not resolved_thread_id:
            print(
                json.dumps(
                    {
                        "error": "Missing thread id for live snapshot. Use --thread-id or --latest-thread.",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            raise typer.Exit(1)
        payload = asyncio.run(
            build_live_state_snapshot(
                thread_id=resolved_thread_id,
                keys=selected_keys,
                include_all=all_keys,
            )
        )
    else:
        payload = build_state_snapshot(keys=selected_keys, include_all=all_keys)
    if not payload and selected_keys:
        print(
            json.dumps(
                {
                    "error": "No valid state keys selected.",
                    "available_keys": list_agent_state_keys(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    _print_section("SNAPSHOT")
    if not payload:
        print("[dim]No state data found.[/dim]")
        return
    for state_key, value in payload.items():
        if isinstance(value, str):
            print(f"[green]{state_key}:[/green]")
            print(value)
            print("")
        else:
            _print_value(state_key, value)


def select_agent_tools() -> list[str]:
    """Select registered tools that this agent is allowed to use."""
    tools = [tool for tool in list_tools() if tool.enabled]

    if not tools:
        print("[yellow]No tools registered yet. Agent will be created without tools.[/yellow]")
        return []

    return inquirer.checkbox(
        message="Select tools this agent can use:",
        choices=[
            {
                "name": f"{tool.tool_key} ({tool.name}) [{tool.type}]",
                "value": tool.tool_key,
            }
            for tool in tools
        ],
    ).execute()


def prompt_provider_for_role(
    *,
    role_label: str,
    default_model: str | None = None,
    default_temperature: float | None = None,
) -> tuple[str, str | None, str, str, float]:
    print(f"\n[bold]{role_label} Provider[/bold]")

    provider_type = inquirer.select(
        message=f"Select {role_label.lower()} provider type:",
        choices=[
            {
                "name": "OpenAI Compatible (OpenAI, local LLMs, etc.)",
                "value": "openai_compatible",
            },
            {
                "name": "Anthropic (Claude)",
                "value": "anthropic",
            },
            {
                "name": "OpenRouter",
                "value": "openrouter",
            },
        ],
        default="openai_compatible",
    ).execute()

    meta = PROVIDER_META[provider_type]
    base_url = (
        typer.prompt(f"{role_label} provider base URL")
        if meta["needs_base_url"]
        else meta["default_base_url"]
    )
    api_key = typer.prompt(f"{role_label} API key", hide_input=True)
    if default_model is None:
        model = typer.prompt(f"{role_label} model")
    else:
        model = typer.prompt(f"{role_label} model", default=default_model)
    if default_temperature is None:
        temperature = typer.prompt(f"{role_label} temperature", type=float)
    else:
        temperature = typer.prompt(
            f"{role_label} temperature",
            type=float,
            default=default_temperature,
        )

    return provider_type, base_url or None, api_key, model, temperature


def create_agent() -> None:
    """Create a new AI agent configuration."""
    name = typer.prompt("Agent display name")
    agent_key = normalize_key(name)

    print(f"[dim]Generated agent key:[/dim] {agent_key}")

    description = typer.prompt("Description", default="")

    (
        decision_provider_type,
        decision_base_url,
        decision_api_key,
        decision_model,
        decision_temperature,
    ) = prompt_provider_for_role(role_label="Decision Maker (Main)")
    (
        lite_provider_type,
        lite_base_url,
        lite_api_key,
        lite_model,
        lite_temperature,
    ) = prompt_provider_for_role(
        role_label="Lite (fast/cheap)",
        default_model=decision_model,
        default_temperature=0.1,
    )
    (
        reviewer_provider_type,
        reviewer_base_url,
        reviewer_api_key,
        reviewer_model,
        reviewer_temperature,
    ) = prompt_provider_for_role(
        role_label="Reviewer",
        default_model=decision_model,
        default_temperature=0.1,
    )
    (
        writer_provider_type,
        writer_base_url,
        writer_api_key,
        writer_model,
        writer_temperature,
    ) = prompt_provider_for_role(
        role_label="Writer (Synthesis)",
        default_model=decision_model,
        default_temperature=0.7,
    )

    print_select_hint(multi=True)
    selected_tools = select_agent_tools()

    try:
        config, path = create_agent_service(
            CreateAgentInput(
                name=name,
                agent_key=agent_key,
                description=description,
                decision_provider_type=decision_provider_type,
                decision_base_url=decision_base_url,
                decision_api_key=decision_api_key,
                decision_model=decision_model,
                decision_temperature=decision_temperature,
                lite_provider_type=lite_provider_type,
                lite_base_url=lite_base_url,
                lite_api_key=lite_api_key,
                lite_model=lite_model,
                lite_temperature=lite_temperature,
                reviewer_provider_type=reviewer_provider_type,
                reviewer_base_url=reviewer_base_url,
                reviewer_api_key=reviewer_api_key,
                reviewer_model=reviewer_model,
                reviewer_temperature=reviewer_temperature,
                writer_provider_type=writer_provider_type,
                writer_base_url=writer_base_url,
                writer_api_key=writer_api_key,
                writer_model=writer_model,
                writer_temperature=writer_temperature,
                selected_tools=selected_tools,
            )
        )
    except (FileExistsError, ValueError) as error:
        print(f"[red]{error}[/red]")
        raise typer.Exit(1)

    print(f"[green]Created agent:[/green] {config.name}")
    print(f"[dim]Key:[/dim] {config.agent_key}")
    print(f"[dim]ID:[/dim] {config.agent_id}")
    print(f"[dim]Tools:[/dim] {', '.join(config.tools) if config.tools else 'None'}")
    print(f"[dim]Saved to:[/dim] {path}")
    print(f"[yellow]Customize persona:[/yellow] {get_persona_path(config.agent_key)}")
    print(f"[yellow]Customize skill:[/yellow] {get_skill_path(config.agent_key)}")
    print(f"[yellow]Customize tools:[/yellow] {get_tools_path(config.agent_key)}")


def list_agents_cmd() -> None:
    """List all configured agents."""
    agents = get_agents()

    if not agents:
        print("[yellow]No agents found.[/yellow]")
        return

    print("\n[bold]Available agents[/bold]\n")

    for agent in agents:
        print(f"[bold cyan]{agent.agent_key}[/bold cyan] ({agent.name})")
        print(f"  [dim]ID:[/dim]    {agent.agent_id}")
        print(f"  [dim]Brain model:[/dim]    {agent.provider.model}")
        print(f"  [dim]Brain temp:[/dim]     {agent.provider.temperature}")
        print(
            f"  [dim]Lite model:[/dim]     "
            f"{agent.lite_provider.model if agent.lite_provider else agent.provider.model}"
        )
        print(
            f"  [dim]Lite temp:[/dim]      "
            f"{agent.lite_provider.temperature if agent.lite_provider else agent.provider.temperature}"
        )
        print(
            f"  [dim]Reviewer model:[/dim] "
            f"{agent.reviewer_provider.model if agent.reviewer_provider else agent.provider.model}"
        )
        print(
            f"  [dim]Reviewer temp:[/dim]  "
            f"{agent.reviewer_provider.temperature if agent.reviewer_provider else agent.provider.temperature}"
        )
        print(
            f"  [dim]Writer model:[/dim]   "
            f"{agent.writer_provider.model if agent.writer_provider else agent.provider.model}"
        )
        print(
            f"  [dim]Writer temp:[/dim]    "
            f"{agent.writer_provider.temperature if agent.writer_provider else agent.provider.temperature}"
        )

        if agent.tools:
            print(f"  [dim]Tools:[/dim] {', '.join(agent.tools)}")
        else:
            print("  [dim]Tools:[/dim] None")

        print("")

