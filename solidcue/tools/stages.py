from __future__ import annotations

from typing import Any, Literal

from solidcue.tools.schema import ToolConfig

ToolStage = Literal["context", "artifact"]

GENERATABLE_TOOL_FIELDS = {"content", "title", "values"}

CONTEXT_TOOL_PREFIXES = (
    "browser_",
    "drive_download_",
    "drive_get_",
    "drive_list_",
    "drive_search_",
    "get_",
    "scrape_",
    "search_",
)

ARTIFACT_TOOL_PREFIXES = (
    "create_",
    "docs_append_",
    "docs_create_",
    "drive_ensure_",
    "drive_move_",
    "drive_upload_",
    "sheets_batch_",
    "sheets_create_",
    "sheets_write_",
)


def infer_tool_stage(tool_key: str, tool_config: ToolConfig | None = None) -> ToolStage:
    if any(tool_key.startswith(prefix) for prefix in CONTEXT_TOOL_PREFIXES):
        return "context"

    if any(tool_key.startswith(prefix) for prefix in ARTIFACT_TOOL_PREFIXES):
        return "artifact"

    if tool_config is not None:
        description = f"{tool_config.name} {tool_config.description}".casefold()
        context_terms = ("search", "list", "get", "download", "scrape", "read", "return")
        if any(term in description for term in context_terms):
            return "context"

        artifact_terms = ("create", "generate", "write", "append", "document", "spreadsheet", "file", "pdf", "csv")
        if any(term in description for term in artifact_terms):
            return "artifact"

    return "context"


def get_required_tool_fields(tool_config: ToolConfig) -> list[str]:
    schema = getattr(getattr(tool_config, "mcp", None), "input_schema", None)
    if not isinstance(schema, dict):
        return []

    required = schema.get("required")
    if not isinstance(required, list):
        return []

    return [field for field in required if isinstance(field, str) and field]


def get_missing_required_tool_fields(
    tool_config: ToolConfig,
    tool_input: dict[str, Any],
) -> list[str]:
    schema = getattr(getattr(tool_config, "mcp", None), "input_schema", None)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    property_map = properties if isinstance(properties, dict) else {}

    missing: list[str] = []
    for field in get_required_tool_fields(tool_config):
        value = tool_input.get(field)
        if value is None:
            missing.append(field)
            continue
        field_schema = property_map.get(field)
        field_type = field_schema.get("type") if isinstance(field_schema, dict) else None
        if field_type == "string" and isinstance(value, str) and not value.strip():
            missing.append(field)

    return missing


def split_missing_tool_fields(fields: list[str]) -> tuple[list[str], list[str]]:
    generatable: list[str] = []
    blocking: list[str] = []

    for field in fields:
        if field in GENERATABLE_TOOL_FIELDS:
            generatable.append(field)
        else:
            blocking.append(field)

    return generatable, blocking
