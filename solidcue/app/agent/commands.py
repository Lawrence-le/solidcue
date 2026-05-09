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

from solidcue.agents.configs.loader import get_persona_path
from solidcue.app.utils.helpers import print_select_hint
from solidcue.app.utils.normalize import normalize_key
from solidcue.core.utils.debug import (
    print_debug_header,
    print_debug_messages,
    print_debug_value,
)
from solidcue.providers.config import PROVIDER_META
from solidcue.services.agent_service import (
    CreateAgentInput,
    create_agent as create_agent_service,
    get_agents,
    run_agent_step as run_agent_step_service,
)
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


def register(app: typer.Typer) -> None:
    app.command("create-agent", rich_help_panel="Agent")(create_agent)
    app.command("list-agents", rich_help_panel="Agent")(list_agents_cmd)
    app.command("run-agent", rich_help_panel="Agent")(run_agent_cmd)


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
) -> tuple[str, str | None, str, str]:
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

    return provider_type, base_url or None, api_key, model


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
    ) = prompt_provider_for_role(role_label="Decision Maker (Main)")
    (
        sufficiency_provider_type,
        sufficiency_base_url,
        sufficiency_api_key,
        sufficiency_model,
    ) = prompt_provider_for_role(
        role_label="Sufficiency Reviewer",
        default_model=decision_model,
    )
    (
        validator_provider_type,
        validator_base_url,
        validator_api_key,
        validator_model,
    ) = prompt_provider_for_role(
        role_label="Validator",
        default_model=decision_model,
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
                sufficiency_provider_type=sufficiency_provider_type,
                sufficiency_base_url=sufficiency_base_url,
                sufficiency_api_key=sufficiency_api_key,
                sufficiency_model=sufficiency_model,
                validator_provider_type=validator_provider_type,
                validator_base_url=validator_base_url,
                validator_api_key=validator_api_key,
                validator_model=validator_model,
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
        print(f"  [dim]Decision model:[/dim]    {agent.provider.model}")
        print(
            f"  [dim]Sufficiency model:[/dim] "
            f"{agent.sufficiency_provider.model if agent.sufficiency_provider else agent.provider.model}"
        )
        print(
            f"  [dim]Validator model:[/dim]   "
            f"{agent.validator_provider.model if agent.validator_provider else agent.provider.model}"
        )

        if agent.tools:
            print(f"  [dim]Tools:[/dim] {', '.join(agent.tools)}")
        else:
            print("  [dim]Tools:[/dim] None")

        print("")


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

    prompt = typer.prompt("Enter your prompt")
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

    messages = result.get("messages", [])
    llm_prompt_messages = result.get("llm_prompt_messages", [])

    if debug:
        print_debug_header("DEBUG Agent Config")
        print(f"Name: {agent.name}")
        print(f"Description: {agent.description}")
        print(f"Provider: {agent.provider.type}")
        print(f"Decision model: {agent.provider.model}")
        print(
            "Sufficiency model: "
            f"{agent.sufficiency_provider.model if agent.sufficiency_provider else agent.provider.model}"
        )
        print(
            "Validator model: "
            f"{agent.validator_provider.model if agent.validator_provider else agent.provider.model}"
        )
        print(f"Tools: {agent.tools}")

        print_debug_header("DEBUG Agent Decision")
        print_debug_value("decision", result.get("decision"))
        print_debug_value("phase", result.get("phase"))
        print_debug_value("router_next", result.get("router_next"))

        print_debug_header("DEBUG Tool Result")
        print_debug_value("execution_result", result.get("execution_result"))
        print_debug_value("context_evidence", result.get("context_evidence"))
        print_debug_value("source_manifest", result.get("source_manifest"))
        print_debug_value("source_evidence", result.get("source_evidence"))
        print_debug_value("artifact_plan", result.get("artifact_plan"))
        print_debug_value("artifact_input", result.get("artifact_input"))
        print_debug_value("artifact_result", result.get("artifact_result"))

        print_debug_header("DEBUG Validation Result")
        print_debug_value("validation_result", result.get("validation_result"))
        print_debug_value("failure_type", result.get("failure_type"))
        print_debug_value("validation_report", result.get("validation_report"))

        print_debug_messages(
            "DEBUG Messages Sent (messages from solidcue/core/graph_node/decision_node.py)",
            messages,
            max_content_len=2500,
            description=(
                "Accumulated user, assistant, and tool transcript for this agent run, "
                "shown in sequence."
            ),
        )
        print_debug_messages(
            "DEBUG Prompt Messages (llm_prompt_messages from solidcue/core/graph_node/decision_node.py)",
            llm_prompt_messages,
            max_content_len=6000,
            description=(
                "Latest decision-model prompt payload, including system instructions, "
                "tool definitions, retry context, and the transcript above."
            ),
        )

    print("\n[green]Response:[/green]")
    _render_user_facing_value(output, panel_multiline=False)
    print()


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
