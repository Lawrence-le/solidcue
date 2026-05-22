import asyncio
import base64
import json
import re
from html.parser import HTMLParser
from typing import Any, cast

from markdown_it import MarkdownIt

from solidcue.agents.configs.loader import load_agent
from solidcue.app.utils.helpers import normalize_tool_output
from solidcue.core.state.schema import AgentState
from solidcue.tools.loader import load_mcp_server, load_tool
from solidcue.tools.mcp.client import MCPClient


TEXT_EXPORT_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
}


def _is_valid_base64(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = re.sub(r"\s+", "", value)
    if not stripped:
        return False
    try:
        base64.b64decode(stripped, validate=True)
        return True
    except Exception:
        return False


def _get_current_task(state: AgentState) -> dict[str, Any] | None:
    task_plan = state.get("task_plan")
    current_task_id = state.get("current_task")
    if not isinstance(task_plan, list) or not current_task_id:
        return None
    return next((t for t in task_plan if isinstance(t, dict) and t.get("id") == current_task_id), None)


def _get_item_key_from_task(task: dict[str, Any] | None) -> str | None:
    if not isinstance(task, dict):
        return None
    context = task.get("context")
    if not isinstance(context, dict):
        return None
    item_key = context.get("item_key")
    if not isinstance(item_key, str):
        return None
    cleaned = item_key.strip()
    return cleaned or None


def _write_handoff(state: AgentState, execution_result: dict[str, Any]) -> dict[str, Any] | None:
    """Store successful task output in the handoff under its requires key."""
    if execution_result.get("success") is not True:
        return None
    task = _get_current_task(state)
    if not task:
        return None
    requires = task.get("requires")
    if not isinstance(requires, list) or not requires:
        return None
    requires_key = str(requires[0])
    content = execution_result.get("content")
    if content is None:
        return None
    handoff = dict(state.get("handoff") or {})
    handoff[requires_key] = content
    item_key = _get_item_key_from_task(task)
    if item_key:
        handoff[f"{requires_key}::{item_key}"] = content
    return handoff


_LARGE_PAYLOAD_FIELDS = {"content_base64", "content", "body", "text", "data"}
_MIN_REAL_CONTENT_LENGTH = 200
_TRUNCATION_MARKER = "… [truncated]"
_TEXT_FALLBACK_FIELD_ORDER = ("content", "text", "body", "data")


def _needs_handoff_fill(value: Any) -> bool:
    """True when the LLM-provided value is likely a placeholder, not real content.

    Large payload fields (content, content_base64, etc.) should contain
    substantial data.  Anything under _MIN_REAL_CONTENT_LENGTH is treated
    as a placeholder or truncated fragment that the handoff should replace.
    Values ending with the prompt-layer truncation marker are always replaced.
    """
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped.endswith(_TRUNCATION_MARKER):
        return True
    return len(stripped) < _MIN_REAL_CONTENT_LENGTH


def _is_truncated_copy(llm_value: str, handoff_value: str) -> bool:
    """True when the LLM value is a prefix of the handoff value.

    The prompt layer truncates expensive fields to ~200 chars.  The LLM
    sometimes copies that truncated fragment (with or without the marker).
    If the handoff holds a longer string that starts with the same bytes,
    the LLM value is almost certainly a truncated copy — use the handoff.
    """
    return (
        len(handoff_value) > len(llm_value)
        and handoff_value.startswith(llm_value.rstrip().removesuffix(_TRUNCATION_MARKER).rstrip())
    )


def _fill_from_handoff(
    arguments: dict[str, Any],
    tool_params: set[str],
    handoff: dict[str, Any],
    excluded_keys: set[str] | None = None,
    force_fill: bool = False,
) -> dict[str, Any]:
    """Inject large payload fields from the handoff into tool arguments.

    Only overrides when the LLM-provided value is missing, is a
    truncation placeholder, or is a truncated copy of the handoff value.
    When the LLM provides real content (e.g. synthesis_draft passed as
    document content), it is preserved.

    Step-by-step behavior:
    1) Determine which tool parameters are eligible for payload fill
       (`tool_params` ∩ `_LARGE_PAYLOAD_FIELDS`).
    2) Build candidate handoff sources from current handoff entries, skipping
       any excluded keys (typically the current task's own output key).
    3) For each eligible target field (e.g., `content`, `content_base64`):
       - Read current argument value from `arguments[field]`.
       - Decide whether replacement is needed via `_needs_handoff_fill(...)`.
    4) Search candidate sources from newest to oldest:
       - Candidate must be a dict and contain the exact same key name
         (`field in dep`).
       - Candidate value must be non-empty (and valid base64 for
         `content_base64`).
    5) Replace only when:
       - current value is placeholder/too short/truncated marker, OR
       - current value is a truncated prefix of the candidate
         (`_is_truncated_copy(...)`).
    6) Return merged arguments for tool execution.

    Note:
    - Matching is exact by key. For target `content`, this function reads
      candidate `content`, not `text`/`body` aliases unless explicitly mapped.
    """
    fields_to_fill = tool_params & _LARGE_PAYLOAD_FIELDS
    if not fields_to_fill:
        return arguments
    merged = dict(arguments)
    blocked = excluded_keys or set()
    candidate_sources: list[Any] = [
        value for key, value in handoff.items() if key not in blocked
    ]

    def _resolve_candidate(dep: dict[str, Any], target_field: str) -> tuple[bool, Any]:
        """Resolve candidate value from handoff entry for a target tool argument.

        Returns:
        - (True, value): source field resolved
        - (False, None): no suitable source in this entry
        """
        # Exact key match remains highest priority.
        if target_field in dep:
            return True, dep[target_field]

        # Text-like payload aliases for non-synthesis source flows:
        # browser_get_html emits `text`, while document tools expect `content`.
        if target_field in {"content", "text", "body", "data"}:
            for source_field in _TEXT_FALLBACK_FIELD_ORDER:
                if source_field in dep:
                    return True, dep[source_field]

        return False, None

    for field in fields_to_fill:
        current_value = merged.get(field)
        needs_fill = _needs_handoff_fill(current_value)
        for dep in reversed(candidate_sources):
            if isinstance(dep, dict):
                found, candidate = _resolve_candidate(dep, field)
                if not found:
                    continue
                if field == "content_base64":
                    if not isinstance(candidate, str):
                        continue
                    normalized_b64 = re.sub(r"\s+", "", candidate)
                    if not _is_valid_base64(normalized_b64):
                        continue
                    candidate = normalized_b64
                if candidate is None:
                    continue
                if isinstance(candidate, str) and not candidate.strip():
                    continue
                # Fill if value is a placeholder OR a truncated copy of the handoff
                if force_fill or needs_fill or (
                    isinstance(current_value, str)
                    and isinstance(candidate, str)
                    and _is_truncated_copy(current_value, candidate)
                ):
                    merged[field] = candidate
                break
    return merged


def _is_artifact_generation_task(state: AgentState, task: dict[str, Any] | None) -> bool:
    """True when current execution is in artifact-generation phase/task."""
    phase = state.get("phase")
    if isinstance(phase, str) and phase.lower() == "artifact":
        return True

    if isinstance(task, dict):
        task_type = task.get("type")
        if isinstance(task_type, str) and task_type.lower() == "artifact_generation":
            return True
        category = task.get("category")
        if isinstance(category, str) and category.lower() == "artifact_generation":
            return True

    return False


def _handoff_for_item(handoff: dict[str, Any], item_key: str | None) -> dict[str, Any]:
    """Prefer item-scoped handoff entries when an item_key is available.

    Entries are scoped by suffix `::<item_key>`. If no scoped entries exist,
    return the full handoff for backward-compatible behavior.
    """
    if not item_key:
        return handoff
    suffix = f"::{item_key}"
    scoped = {k: v for k, v in handoff.items() if isinstance(k, str) and k.endswith(suffix)}
    return scoped or handoff


def _execution_result(success: bool, result_type: str, content: Any, error: Any) -> dict[str, Any]:
    return {
        "success": success,
        "type": result_type,
        "content": content,
        "error": error,
    }


def _record_tool_call(
    state: AgentState,
    success: bool,
    execution_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    decision = state.get("active_tool_call")
    if not isinstance(decision, dict):
        return []

    tool_name = decision.get("tool_name")
    tool_input = decision.get("tool_input")
    if not isinstance(tool_name, str) or not tool_name:
        return []

    history = state.get("tool_call_history")
    normalized_history = history if isinstance(history, list) else []
    normalized_tool_input = tool_input if isinstance(tool_input, dict) else {}
    task_id = state.get("current_task")

    effective_execution_result = execution_result if isinstance(execution_result, dict) else (state.get("execution_result") or {})
    return [
        *normalized_history,
        {
            "task_id": str(task_id) if isinstance(task_id, str) and task_id else None,
            "tool_name": tool_name,
            "tool_input": normalized_tool_input,
            "success": success,
            "execution_result": effective_execution_result if isinstance(effective_execution_result, dict) else None,
        },
    ]


def _is_text_mime_type(mime_type: Any) -> bool:
    if not isinstance(mime_type, str):
        return False

    normalized = mime_type.split(";", maxsplit=1)[0].strip().lower()
    return normalized.startswith("text/") or normalized in TEXT_EXPORT_MIME_TYPES or normalized.endswith("+json")


def _decode_text_bytes(raw: bytes) -> str:
    return raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n").strip()


def _decode_file_content(content: Any) -> Any:
    """Normalize tool output: parse JSON strings to dicts and decode base64 file content.

    normalize_tool_output may return a JSON string (via MCP text content items).
    This function ensures callers always receive a dict so field access works
    uniformly — whether checking for documentId, encoding, or file text.
    """
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped.startswith("{"):
            return content
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return content
        if not isinstance(parsed, dict):
            return content
    elif isinstance(content, dict):
        parsed = content
    else:
        return content

    if parsed.get("encoding") == "base64":
        raw = parsed.get("content")
        if isinstance(raw, str) and raw.strip():
            try:
                decoded_bytes = base64.b64decode(raw)
                mime_type = parsed.get("mimeType") or parsed.get("mime_type")
                if mime_type and not _is_text_mime_type(mime_type):
                    return parsed

                decoded = _decode_text_bytes(decoded_bytes)
                result = {k: v for k, v in parsed.items() if k not in ("encoding", "content")}
                result["content"] = decoded
                return result
            except Exception:
                pass

    return parsed





# ======================================================================================
#                          WEB CONTENT CLEANING
# Cleans scraped web content immediately after tool execution so that all downstream
# nodes (reflection, synthesis) always receive clean, noise-free content.
#
# Flow:
#   _is_web_content   — detects if tool output is scraped web content
#   _HTMLTextExtractor — strips HTML tags from raw HTML output
#   _strip_html        — convenience wrapper around _HTMLTextExtractor
#   _clean_text        — removes boilerplate lines + normalises markdown
#   _clean_content     — orchestrates detection + cleaning on tool output
# ======================================================================================

_WEB_CONTENT_KEYS = {"text", "html", "markdown"}

_NOISE_LINE_RE = re.compile(
    r"^(skip to|sign in|join now|log in|sign up|cookie|subscribe|advertisement|"
    r"see who|apply|save|show more|show less|\d+\s+applicants?|clear text|"
    r"use ai|am i a good fit|tailor my resume|get ai[- ]powered)",
    re.IGNORECASE,
)

_md = MarkdownIt()


def _is_web_content(content: Any) -> bool:
    """Detect scraped web content by checking for common scraper output keys."""
    if isinstance(content, dict):
        return bool(_WEB_CONTENT_KEYS & content.keys())
    if isinstance(content, list) and content:
        return all(isinstance(item, dict) and bool(_WEB_CONTENT_KEYS & item.keys()) for item in content)
    return False


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and collect visible text."""
    _SKIP_TAGS = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _strip_html(text: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(text)
    return extractor.get_text()


def _clean_text(text: str) -> str:
    """Strip HTML/boilerplate from scraped web content — zero LLM cost."""
    if re.search(r"<[a-zA-Z][^>]*>", text):
        text = _strip_html(text)
    else:
        tokens = _md.parse(text)
        plain_parts: list[str] = []
        for token in tokens:
            if token.type == "inline" and token.children:
                for child in token.children:
                    if child.type == "text" and child.content:
                        plain_parts.append(child.content)
            elif token.content:
                plain_parts.append(token.content)
        if plain_parts:
            text = "\n".join(plain_parts)

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 4:
            continue
        if _NOISE_LINE_RE.match(stripped):
            continue
        lines.append(stripped)

    result = "\n".join(lines)
    return result or text


def _clean_content(content: Any) -> Any:
    """Clean web content and strip raw web keys. Non-web content is returned as-is."""
    if not _is_web_content(content):
        return content

    if isinstance(content, list):
        cleaned = []
        for item in content:
            raw = str(item.get("html") or item.get("text") or item.get("markdown") or item.get("content") or "").strip()
            if raw:
                meta = {k: v for k, v in item.items() if k not in _WEB_CONTENT_KEYS}
                cleaned.append({**meta, "text": _clean_text(raw)})
        return cleaned

    if isinstance(content, dict):
        raw = str(content.get("html") or content.get("text") or content.get("markdown") or content.get("content") or "").strip()
        meta = {k: v for k, v in content.items() if k not in _WEB_CONTENT_KEYS}
        return {**meta, "text": _clean_text(raw)}

    return content


# ======================================================================================


def _execute_tool(state: AgentState) -> dict[str, Any]:
    decision = cast(dict[str, Any], state.get("active_tool_call") or {})

    action = decision.get("action")

    if action != "use_tool":
        return {
            "execution_result": _execution_result(
                success=True,
                result_type="skipped",
                content=None,
                error=None,
            )
        }

    tool_key = decision.get("tool_name")
    raw_arguments = decision.get("tool_input")
    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}

    handoff = state.get("handoff")
    if isinstance(handoff, dict) and handoff:
        try:
            tool_cfg = load_tool(tool_key)
            schema = getattr(getattr(tool_cfg, "mcp", None), "input_schema", None)
            tool_params = set(schema.get("properties", {}).keys()) if isinstance(schema, dict) else set()
            task = _get_current_task(state)
            item_key = _get_item_key_from_task(task)
            requires = task.get("requires") if isinstance(task, dict) else None
            current_output_keys = {str(item) for item in requires if isinstance(item, str)} if isinstance(requires, list) else set()
            handoff_view = _handoff_for_item(handoff, item_key)
            arguments = _fill_from_handoff(
                arguments,
                tool_params,
                handoff_view,
                excluded_keys=current_output_keys,
                force_fill=_is_artifact_generation_task(state, task),
            )
        except Exception:
            pass
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {
            "execution_result": _execution_result(
                success=False,
                result_type="tool_execution",
                content=None,
                error="Execution Error: agent_key missing",
            )
        }

    try:
        agent = load_agent(agent_key)
        if tool_key not in set(agent.tools or []):
            raise ValueError(f"Tool '{tool_key}' not allowed for agent '{agent_key}'")

        selected_tool = load_tool(tool_key)
        if selected_tool.type != "mcp" or not selected_tool.mcp:
            raise ValueError(f"Unsupported or misconfigured tool type: {selected_tool.type}")

        server = load_mcp_server(selected_tool.mcp.server_key)
        client = MCPClient(server)

        raw_output = asyncio.run(
            client.call_tool(
                tool_name=selected_tool.mcp.tool_name,
                arguments=arguments,
            )
        )

        normalized_output = _decode_file_content(normalize_tool_output(raw_output))
        is_tool_error = bool(raw_output.get("is_error"))
        if not is_tool_error:
            normalized_output = _clean_content(normalized_output)
        if is_tool_error and isinstance(normalized_output, str) and "Unable to reach Open-Meteo service" in normalized_output:
            normalized_output = (
                f"{normalized_output}. The MCP server is running, but its outbound network to Open-Meteo failed."
            )

        update: dict[str, Any] = {
            "execution_result": _execution_result(
                success=not is_tool_error,
                result_type="tool_execution",
                content=normalized_output,
                error=normalized_output if is_tool_error else None,
            )
        }

        return update

    except Exception as exc:
        error_text = str(exc)
        if "Unable to reach MCP server" in error_text:
            error_text = f"{error_text}. Check that the MCP service is running and reachable."

        return {
            "execution_result": _execution_result(
                success=False,
                result_type="tool_execution",
                content=None,
                error=f"Execution Error: {error_text}",
            )
        }


def execution_node(state: AgentState) -> dict[str, Any]:
    update: dict[str, Any] = {}

    execution_update = _execute_tool(state)
    update.update(execution_update)

    tool_turn_count_value = state.get("tool_turn_count")
    tool_turn_count = tool_turn_count_value if isinstance(tool_turn_count_value, int) else 0
    update["tool_turn_count"] = tool_turn_count + 1

    execution_result = execution_update.get("execution_result")
    succeeded = isinstance(execution_result, dict) and execution_result.get("success") is True
    update["tool_call_history"] = _record_tool_call(
        state,
        success=succeeded,
        execution_result=execution_result if isinstance(execution_result, dict) else None,
    )

    if isinstance(execution_result, dict):
        updated_handoff = _write_handoff(state, execution_result)
        if updated_handoff is not None:
            update["handoff"] = updated_handoff

    return update
