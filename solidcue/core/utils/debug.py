import json
import re
from typing import Any, Mapping

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

console = Console()
CLI_DEBUG_HEADER_STYLE = "bold cyan"
WORKFLOW_DEBUG_HEADER_STYLE = "bold magenta"

NODE_KEYS = {
    "decision_node": [
        "metadata",
        "phase",
        "source_manifest",
        "decision",
        "active_tool_call",
        "artifact_plan",
    ],
    "execution_node": [
        "messages",
        "execution_result",
        "context_evidence",
        "source_manifest",
        "source_evidence",
    ],
    "router_node": ["phase", "failure_type", "router_next"],
    "artifact_generation_node": ["artifact_input", "artifact_generation_messages"],
    "artifact_execution_node": ["artifact_result"],
    "post_execution_reflection_node": ["reflection_result", "tool_turn_count", "tool_call_history"],
    "validation_node": ["validation_result", "failure_type", "validation_report"],
    "synthesis_node": ["synthesis_draft"],
    "final_output_node": ["final_output", "final_response"],
}

NODE_STATE_KEY_STYLE = "green"


_SENSITIVE_QUERY_PARAM_RE = re.compile(r"([?&](?:api_key|key|token|access_token)=)[^&\s]+", re.IGNORECASE)
_SENSITIVE_JSON_FIELD_RE = re.compile(
    r'("?(?:api_key|key|token|access_token)"?\s*[:=]\s*")([^"]+)(")',
    re.IGNORECASE,
)


def preview(v: Any, max_len: int = 150) -> Any:
    if isinstance(v, str) and len(v) > max_len:
        return v[:max_len] + "..."
    return v


def sanitize_debug_text(text: str) -> str:
    text = _SENSITIVE_QUERY_PARAM_RE.sub(r"\1[redacted]", text)
    return _SENSITIVE_JSON_FIELD_RE.sub(r"\1[redacted]\3", text)


def truncate_debug_text(text: str, max_len: int = 4000) -> str:
    if len(text) <= max_len:
        return text

    omitted = len(text) - max_len
    return f"{text[:max_len]}\n... [truncated {omitted} chars]"


def _parse_json_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_debug_value(value: Any) -> Any:
    if isinstance(value, str):
        parsed = _parse_json_string(value)
        if parsed is value:
            return value
        return _normalize_debug_value(parsed)

    if isinstance(value, list):
        return [_normalize_debug_value(item) for item in value]

    if isinstance(value, dict):
        return {key: _normalize_debug_value(item) for key, item in value.items()}

    return value


def format_debug_value(value: Any, max_len: int = 4000) -> str:
    normalized = _normalize_debug_value(value)

    if isinstance(normalized, str):
        return truncate_debug_text(sanitize_debug_text(normalized), max_len=max_len)

    try:
        formatted = json.dumps(normalized, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        formatted = str(normalized)

    return truncate_debug_text(sanitize_debug_text(formatted), max_len=max_len)


def print_debug_header(title: str, *, style: str = CLI_DEBUG_HEADER_STYLE) -> None:
    console.print(f"\n[{style}]{title}[/{style}]")


def print_debug_separator(title: str, description: str | None = None) -> None:
    console.print(f"\n[dim]──────────────── {title} ────────────────[/dim]")
    if description:
        console.print(f"[dim]{description}[/dim]")


def print_debug_value(label: str, value: Any, max_len: int = 4000) -> None:
    formatted = format_debug_value(value, max_len=max_len)

    if isinstance(value, (dict, list)):
        console.print(f"[bold bright_white]{label}[/bold bright_white]:")
        console.print(Text(formatted))
        return

    console.print(f"[bold bright_white]{label}[/bold bright_white]: {formatted}")


def _message_title(index: int, message: Mapping[str, Any]) -> str:
    role = str(message.get("role", "unknown"))
    tool_calls = message.get("tool_calls")

    if isinstance(tool_calls, list) and tool_calls:
        first_call = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        function_info = first_call.get("function") if isinstance(first_call, dict) else {}
        function_name = function_info.get("name") if isinstance(function_info, dict) else None
        if function_name:
            return f"{index}. {role} -> tool_call:{function_name}"

    if role == "tool":
        tool_call_id = message.get("tool_call_id", "unknown")
        return f"{index}. tool result:{tool_call_id}"

    return f"{index}. {role}"


def _message_body(message: Mapping[str, Any], max_len: int) -> str:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return format_debug_value(tool_calls, max_len=max_len)

    content = message.get("content")
    if message.get("role") == "tool":
        return format_debug_value(content, max_len=max_len)

    return format_debug_value("" if content is None else str(content), max_len=max_len)


def print_debug_messages(
    title: str,
    messages: Any,
    *,
    max_content_len: int = 4000,
    header_style: str = CLI_DEBUG_HEADER_STYLE,
    description: str | None = None,
) -> None:
    print_debug_header(title, style=header_style)
    if description:
        console.print(f"[dim]{description}[/dim]")

    if not isinstance(messages, list) or not messages:
        console.print("[dim]None[/dim]")
        return

    for idx, message in enumerate(messages, start=1):
        if not isinstance(message, Mapping):
            console.print(Panel(Text(format_debug_value(message, max_len=max_content_len)), title=str(idx)))
            continue

        body = _message_body(message, max_len=max_content_len)
        console.print(
            Panel(
                Text(body),
                title=_message_title(idx, message),
                title_align="left",
                border_style="dim",
            )
        )


def _format_node_state_line(key: str, value: Any) -> Text:
    line = Text()
    line.append(key, style=NODE_STATE_KEY_STYLE)
    line.append(": ")
    line.append(format_debug_value(preview(value), max_len=1200))
    return line


def log_state(node_name: str, state: Mapping[str, Any]) -> None:
    keys = NODE_KEYS.get(node_name, state.keys())
    rendered_values = []

    for k in keys:
        if k in state:
            rendered_values.append(_format_node_state_line(k, state[k]))

    body = Group(*rendered_values) if rendered_values else Text("No selected state fields.", style="dim")
    console.print(
        Panel(
            body,
            title=f"NODE {node_name}",
            title_align="left",
            border_style=WORKFLOW_DEBUG_HEADER_STYLE,
        )
    )
