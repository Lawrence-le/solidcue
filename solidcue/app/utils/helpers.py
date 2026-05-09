
from rich import print

def print_select_hint(multi: bool = False) -> None:
    if multi:
        print("[dim]Use ↑/↓ to navigate • [bold]Space[/bold] to select • [bold]Enter[bold] to confirm[/dim]")
    else:
        print("[dim]Use ↑/↓ to navigate • [bold]Enter[bold] to select[/dim]")


def normalize_tool_output(raw):
    if isinstance(raw, dict):
        content = raw.get("content")

        if isinstance(content, list) and content:
            first = content[0]

            if isinstance(first, dict):
                return first.get("text") or first.get("content") or first

            return first

        return raw

    if isinstance(raw, str):
        return raw

    return raw
