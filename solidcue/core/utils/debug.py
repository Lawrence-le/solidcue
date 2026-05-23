import json
import re
from typing import Any, Mapping

from rich.console import Console, Group
from rich.text import Text
from rich.panel import Panel

console = Console()
CLI_DEBUG_HEADER_STYLE = "bold cyan"
WORKFLOW_DEBUG_HEADER_STYLE = "bold magenta"

NODE_STATE_KEY_STYLE = "green"


_SENSITIVE_QUERY_PARAM_RE = re.compile(r"([?&](?:api_key|key|token|access_token)=)[^&\s]+", re.IGNORECASE)
_SENSITIVE_JSON_FIELD_RE = re.compile(
    r'("?(?:api_key|access_token|token|secret|client_secret|refresh_token|authorization)"?\s*[:=]\s*")([^"]+)(")',
    re.IGNORECASE,
)


def preview(v: Any, max_len: int = 150) -> Any:
    if isinstance(v, str) and len(v) > max_len:
        return v[:max_len] + "..."
    return v


def sanitize_debug_text(text: str) -> str:
    text = _SENSITIVE_QUERY_PARAM_RE.sub(r"\1[redacted]", text)
    return _SENSITIVE_JSON_FIELD_RE.sub(r"\1[redacted]\3", text)


def truncate_debug_text(text: str, max_len: int = 1000) -> str:
    if len(text) <= max_len:
        return text

    omitted = len(text) - max_len
    return f"{text[:max_len]}\n... [truncated {omitted} chars]"

# def truncate_debug_text(text: str, max_len: int = 4000) -> str:
#     # Temporary: disable truncation so full debug payload is visible.
#     return text

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


_UNTRUNCATED_STATE_KEYS = {"tool_call_history"}
_TRUNCATED_TEXT_KEYS = {"text"}
_VERBOSE_STATE_KEYS: set[str] = set()


def _truncate_text_content_fields(value: Any, *, max_len: int = 100) -> Any:
    if isinstance(value, list):
        return [_truncate_text_content_fields(item, max_len=max_len) for item in value]

    if isinstance(value, dict):
        truncated: dict[Any, Any] = {}
        for key, item in value.items():
            key_lower = key.lower() if isinstance(key, str) else ""
            should_truncate = key_lower in _TRUNCATED_TEXT_KEYS or "content" in key_lower
            if should_truncate and isinstance(item, str):
                truncated[key] = preview(item, max_len=max_len)
            else:
                truncated[key] = _truncate_text_content_fields(item, max_len=max_len)
        return truncated

    return value


def _format_node_state_line(key: str, value: Any) -> Text:
    line = Text()
    line.append(key, style=NODE_STATE_KEY_STYLE)
    line.append(": ")
    if key in _UNTRUNCATED_STATE_KEYS:
        if key == "tool_call_history":
            value = _truncate_text_content_fields(value, max_len=100)
        line.append(format_debug_value(value, max_len=999999))
        return line

    if key in _VERBOSE_STATE_KEYS:
        # Show last 1000 chars of the serialized value so content is visible
        serialized = format_debug_value(value, max_len=999999)
        snippet = serialized[-1000:] if len(serialized) > 1000 else serialized
        line.append(snippet)
    else:
        line.append(format_debug_value(preview(value), max_len=1200))
    return line


def log_state(node_name: str, state: Mapping[str, Any]) -> None:
    rendered_values = []

    for k in state.keys():
        if k == "metric_usage_events":
            continue
        if node_name == "decision" and k == "llm_prompt_messages":
            continue
        if node_name == "router" and k == "task_plan":
            continue
        if node_name == "execution" and k == "tool_call_history":
            continue
        if node_name == "router" and k == "retry_reason":
            line = Text()
            line.append(k, style=NODE_STATE_KEY_STYLE)
            line.append(": ")
            line.append(format_debug_value(state[k], max_len=999999))
            rendered_values.append(line)
            continue
        if node_name == "planning" and k == "task_plan":
            line = Text()
            line.append(k, style=NODE_STATE_KEY_STYLE)
            line.append(": ")
            line.append(format_debug_value(state[k], max_len=999999))
            rendered_values.append(line)
            continue
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
