import typer
from InquirerPy import inquirer
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme
from typing import Any
import json

from solidcue.agent_configs.loader import get_persona_path, get_skill_path, get_tools_path
from solidcue.app.utils.helpers import print_select_hint
from solidcue.app.utils.normalize import normalize_key
from solidcue.providers.config import PROVIDER_META
from solidcue.services.agent_service import CreateAgentInput, create_agent as create_agent_service
from solidcue.services.run_engine import run_agent_step as run_agent_step_service
from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    build_state_snapshot,
    get_latest_thread_id,
    list_agent_state_keys,
)
from solidcue.services.workspace_service import get_agents
from solidcue.services.thread_service import create_thread_id
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
            resolved_thread_id = get_latest_thread_id()
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
        payload = build_live_state_snapshot(
            thread_id=resolved_thread_id,
            keys=selected_keys,
            include_all=all_keys,
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


def _read_multiline_prompt() -> str:
    """Read a multi-line prompt from the user.

    Supports two input modes:
    1. Single-line: type a message and press Enter — returns immediately.
    2. Multi-line paste: detects buffered text and drains it all.

    Uses a two-phase timeout strategy:
    - Short initial probe (50ms) after the first line to detect if this is
      a paste or a typed single line.
    - Longer drain timeout (500ms) between subsequent lines to tolerate
      chunked paste delivery from the terminal.

    This prevents leftover pasted text from leaking into the shell after
    the application exits.
    """
    import sys
    import select

    print("[dim]Enter your prompt (paste multi-line text freely, end with an empty line):[/dim]")
    lines: list[str] = []
    first_line = sys.stdin.readline()
    if not first_line:
        return ""
    lines.append(first_line.rstrip("\n").rstrip("\r"))

    # Short probe: is there more data buffered (paste) or was this typed?
    if not hasattr(select, "select"):
        return "\n".join(lines).strip()

    ready, _, _ = select.select([sys.stdin], [], [], 0.05)
    if not ready:
        # Single-line typed input — return immediately.
        return "\n".join(lines).strip()

    # Multi-line paste detected — drain with a generous timeout to handle
    # chunked delivery. Terminal emulators paste in bursts with gaps that
    # can exceed 100ms on large payloads.
    while True:
        line = sys.stdin.readline()
        if not line:  # EOF
            break
        lines.append(line.rstrip("\n").rstrip("\r"))
        # Wait for next chunk; 500ms tolerates slow paste delivery.
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if not ready:
            break

    return "\n".join(lines).strip()


def run_agent_cmd(debug: bool = False) -> None:
    """Run an existing AI agent."""
    agents = get_agents()

    if not agents:
        print("[red]No agents found.[/red]")
        raise typer.Exit(1)

    agent_key = inquirer.select(
        message="Select agent:",
        choices=[
            {
                "name": f"{agent.agent_key} ({agent.name})",
                "value": agent.agent_key,
            }
            for agent in agents
        ],
    ).execute()

    prompt = _read_multiline_prompt()
    thread_id = create_thread_id()

    try:
        agent, result = run_agent_step_service(
            agent_key=agent_key,
            user_input=prompt,
            thread_id=thread_id,
            debug=debug,
        )
        while True:
            payload = _extract_interrupt_payload(result)
            if payload is None:
                break
            resume_value = _prompt_interrupt_resume(payload)
            _, result = run_agent_step_service(
                agent_key=agent_key,
                thread_id=thread_id,
                debug=debug,
                resume_value=resume_value,
            )
    except Exception as error:
        print(f"[red]{error}[/red]")
        raise typer.Exit(1)

    if debug:
        print("[dim]──────────────── CLI debug summary ────────────────[/dim]")

    print(f"[yellow]Running agent:[/yellow] {agent.name}")

    output = (
        result.get("final_output")
        or result.get("final_response")
        or result.get("synthesis_draft")
        or result.get("draft_output")
        or "No final response generated."
    )

    if debug:
        _print_section("DEBUG Agent Config")
        print(f"Name: {agent.name}")
        print(f"Description: {agent.description}")
        print(f"Provider: {agent.provider.type}")
        print(f"Brain model: {agent.provider.model}")
        print(f"Brain temp: {agent.provider.temperature}")
        print(
            "Lite model: "
            f"{agent.lite_provider.model if agent.lite_provider else agent.provider.model}"
        )
        print(
            "Lite temp: "
            f"{agent.lite_provider.temperature if agent.lite_provider else agent.provider.temperature}"
        )
        print(
            "Reviewer model: "
            f"{agent.reviewer_provider.model if agent.reviewer_provider else agent.provider.model}"
        )
        print(
            "Reviewer temp: "
            f"{agent.reviewer_provider.temperature if agent.reviewer_provider else agent.provider.temperature}"
        )
        print(
            "Writer model: "
            f"{agent.writer_provider.model if agent.writer_provider else agent.provider.model}"
        )
        print(
            "Writer temp: "
            f"{agent.writer_provider.temperature if agent.writer_provider else agent.provider.temperature}"
        )
        print(f"Tools: {agent.tools}")
        _print_metric_usage_summary(result)

    print("\n[green]Response:[/green]")
    _render_user_facing_value(output, panel_multiline=False)
    print()


def _print_metric_usage_summary(result: dict[str, Any]) -> None:
    _NODE_EXECUTION_ORDER = [
        "initialize",
        "classifier",
        "discovery",
        "planning",
        "decision",
        "execution",
        "reflection",
        "synthesis",
        "validation",
        "validation_hhem",
        "final_output",
    ]
    _node_order_index = {name: i for i, name in enumerate(_NODE_EXECUTION_ORDER)}

    metric_key_to_node = {
        "metric_classifier": "classifier",
        "metric_discovery": "discovery",
        "metric_planning": "planning",
        "metric_decision": "decision",
        "metric_reflection": "reflection",
        "metric_synthesis": "synthesis",
        "metric_validation": "validation",
        "metric_validation_hhem": "validation_hhem",
        "metric_final_output": "final_output",
    }

    aggregated_by_node: dict[str, dict[str, Any]] = {}
    events = result.get("metric_usage_events")
    if isinstance(events, list) and events:
        for event in events:
            if not isinstance(event, dict):
                continue
            node_name = str(event.get("node") or "").strip()
            if not node_name:
                continue
            if node_name not in aggregated_by_node:
                aggregated_by_node[node_name] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "estimated_total": 0,
                    "estimated_system": 0,
                    "estimated_user": 0,
                    "estimated_assistant": 0,
                    "llm_call_count": 0,
                    "message_count": 0,
                    "time_s": 0.0,
                    "models": set(),
                    "methods": set(),
                }
            acc = aggregated_by_node[node_name]
            acc["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
            acc["completion_tokens"] += int(event.get("completion_tokens") or 0)
            acc["total_tokens"] += int(event.get("total_tokens") or 0)
            acc["cached_tokens"] += int(event.get("cached_tokens") or 0)
            acc["estimated_total"] += int(event.get("estimated_total") or 0)
            acc["estimated_system"] += int(event.get("estimated_system") or 0)
            acc["estimated_user"] += int(event.get("estimated_user") or 0)
            acc["estimated_assistant"] += int(event.get("estimated_assistant") or 0)
            acc["llm_call_count"] += int(event.get("llm_call_count") or 0)
            acc["message_count"] += int(event.get("message_count") or 0)
            acc["time_s"] += float(event.get("time_s") or 0.0)
            model = str(event.get("model") or "").strip()
            if model:
                acc["models"].add(model)
            method = str(event.get("method") or "").strip()
            if method:
                acc["methods"].add(method)

    rows: list[tuple[str, dict[str, Any]]] = []
    if aggregated_by_node:
        rows = sorted(
            aggregated_by_node.items(),
            key=lambda item: _node_order_index.get(item[0], 999),
        )
    else:
        for metric_key, node_name in metric_key_to_node.items():
            payload = result.get(metric_key)
            if isinstance(payload, dict) and payload:
                token_payload = payload.get("tokens")
                if isinstance(token_payload, dict):
                    merged_payload = dict(token_payload)
                    merged_payload["time_s"] = float(payload.get("time_s") or 0.0)
                    merged_payload["model"] = str(payload.get("model") or "")
                    rows.append((node_name, merged_payload))

    _print_section("DEBUG Metric Summary")
    if not rows:
        print("[dim]No token usage data found in final state.[/dim]")
        return

    table = Table(show_header=True, header_style="bold bright_white")
    table.add_column("Node", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Token Source", style="yellow")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Cached", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("Messages", justify="right")
    table.add_column("Time (s)", justify="right")

    grand_total = 0
    for node_name, payload in rows:
        prompt_tokens = int(payload.get("prompt_tokens") or 0)
        completion_tokens = int(payload.get("completion_tokens") or 0)
        total_tokens = int(payload.get("total_tokens") or 0)
        cached_tokens = int(payload.get("cached_tokens") or 0)
        llm_call_count = int(
            payload.get("llm_call_count")
            or (1 if total_tokens > 0 or int(payload.get("estimated_total") or 0) > 0 else 0)
        )
        message_count = int(payload.get("message_count") or 0)
        time_s = float(payload.get("time_s") or 0.0)
        models = payload.get("models")
        if isinstance(models, set):
            model = ", ".join(sorted(models)) if models else "-"
        else:
            model = str(payload.get("model") or "-")
        methods = payload.get("methods")
        if isinstance(methods, set):
            if "provider_reported" in methods:
                source = "provider_reported"
            elif methods:
                source = ",".join(sorted(methods))
            else:
                source = "-"
        else:
            method = str(payload.get("method") or "").strip()
            source = method if method else "-"
        grand_total += total_tokens
        table.add_row(
            node_name,
            model,
            source,
            str(prompt_tokens),
            str(completion_tokens),
            str(total_tokens),
            str(cached_tokens),
            str(llm_call_count),
            str(message_count),
            f"{time_s:.3f}",
        )

    table.add_section()
    total_time_s = sum(float(payload.get("time_s") or 0.0) for _, payload in rows)
    table.add_row(
        "TOTAL",
        "-",
        "-",
        "-",
        "-",
        str(grand_total),
        "-",
        "-",
        "-",
        f"{total_time_s:.3f}",
    )
    console.print(table)


def _extract_interrupt_payload(result: Any) -> dict[str, Any] | None:
    # v1 dict-style interrupt envelope
    if isinstance(result, dict):
        raw_interrupts = result.get("__interrupt__")
        if isinstance(raw_interrupts, (list, tuple)) and raw_interrupts:
            first = raw_interrupts[0]
            value = getattr(first, "value", None)
            if isinstance(value, dict):
                return value
        return None

    # v2 GraphOutput style
    raw_interrupts = getattr(result, "interrupts", None)
    if isinstance(raw_interrupts, (list, tuple)) and raw_interrupts:
        first = raw_interrupts[0]
        value = getattr(first, "value", None)
        if isinstance(value, dict):
            return value
    return None


def _prompt_interrupt_resume(payload: dict[str, Any]) -> str:
    mode = payload.get("mode")
    prompt = payload.get("prompt") or "Approval required."
    preview = payload.get("preview")

    print("\n[bold yellow]Approval Required[/bold yellow]")
    print(str(prompt))
    if isinstance(preview, dict):
        title = preview.get("title")
        summary = preview.get("summary")
        sections = preview.get("sections")
        if isinstance(title, str) and title:
            print(f"\n[bold]{title}[/bold]")
        if isinstance(summary, str) and summary:
            print(summary)
        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue
                label = section.get("label")
                content = section.get("content")
                if isinstance(label, str) and isinstance(content, str):
                    print(f"\n[cyan]{label}:[/cyan]")
                    if label.strip().lower() == "tool input":
                        parsed = _try_parse_structured(content)
                        if isinstance(parsed, (dict, list)):
                            print(
                                Panel(
                                    Syntax(
                                        json.dumps(parsed, ensure_ascii=False, indent=2, default=str),
                                        "json",
                                        word_wrap=True,
                                    )
                                )
                            )
                            continue
                    _render_user_facing_value(content, panel_multiline=True)

    options = payload.get("options")
    if mode == "deterministic" and isinstance(options, list) and options:
        normalized_options = [str(option).upper() for option in options]
        return inquirer.select(
            message="Choose approval action:",
            choices=[{"name": option, "value": option} for option in normalized_options],
        ).execute()

    return typer.prompt("Enter response")


def _try_parse_structured(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] not in "{[":
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _render_dict_table(data: dict[str, Any]) -> None:
    table = Table(show_header=True, header_style="bold bright_white", expand=True)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value")
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        else:
            value_text = str(value)
        table.add_row(str(key), value_text)
    print(table)


def _render_list_table(data: list[Any]) -> bool:
    if not data:
        print("[]")
        return True
    if not all(isinstance(item, dict) for item in data):
        return False

    dict_rows = [item for item in data if isinstance(item, dict)]
    headers: list[str] = []
    seen: set[str] = set()
    for row in dict_rows:
        for key in row.keys():
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                headers.append(key_str)

    if not headers:
        return False

    table = Table(show_header=True, header_style="bold bright_white", expand=True)
    for header in headers:
        table.add_column(header)

    for row in dict_rows:
        rendered_row: list[str] = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, (dict, list)):
                rendered_row.append(json.dumps(value, ensure_ascii=False, default=str))
            elif value is None:
                rendered_row.append("")
            else:
                rendered_row.append(str(value))
        table.add_row(*rendered_row)

    print(table)
    return True


def _render_user_facing_value(value: Any, *, panel_multiline: bool = False) -> None:
    parsed = _try_parse_structured(value)
    if isinstance(parsed, dict):
        _render_dict_table(parsed)
        return
    if isinstance(parsed, list):
        if _render_list_table(parsed):
            return
        print(Panel(json.dumps(parsed, ensure_ascii=False, indent=2, default=str)))
        return

    if panel_multiline and isinstance(value, str) and "\n" in value:
        print(Panel(Markdown(value)))
        return

    if isinstance(value, str):
        console.print(Markdown(value))
        return

    print("" if value is None else str(value))
