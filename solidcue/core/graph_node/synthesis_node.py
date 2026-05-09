import json
from typing import Any, cast

from solidcue.agents.configs.loader import load_agent, load_agent_persona
from solidcue.core.state.schema import AgentState
from solidcue.prompts.synthesis_prompt import build_synthesis_messages
from solidcue.providers.factory import get_provider


def _parse_json_if_possible(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _truncate(text: Any, limit: int = 180) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1].rstrip()}…"


def _format_result_item(item: Any, index: int) -> str:
    if not isinstance(item, dict):
        return f"{index}. {_truncate(item)}"

    title = _truncate(item.get("title") or "Untitled")
    date = str(item.get("date") or "").strip()
    url = str(item.get("url") or "").strip()
    snippet = _truncate(item.get("snippet") or "", 220)

    parts: list[str] = [f"{index}. {title}"]
    if date:
        parts.append(f"Date: {date}")
    if url:
        parts.append(f"URL: {url}")
    if snippet:
        parts.append(f"Summary: {snippet}")
    return "\n".join(parts)


def _summarize_structured_tool_output(content: Any) -> str | None:
    parsed = _parse_json_if_possible(content)

    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        query = str(parsed.get("query") or "").strip()
        rows = parsed.get("results") or []
        if not rows:
            return None

        top_n = rows[:5]
        if query:
            header = f"Query: {query}\nTop results ({len(top_n)} of {len(rows)}):"
        else:
            header = f"Top results ({len(top_n)} of {len(rows)}):"

        lines = [header, ""]
        for idx, row in enumerate(top_n, start=1):
            lines.append(_format_result_item(row, idx))
            if idx < len(top_n):
                lines.append("")
        return "\n".join(lines)

    if isinstance(parsed, dict) and parsed:
        # Convert generic structured payloads (e.g., weather dicts) into
        # readable text so synthesis does not inherit raw JSON formatting.
        lines = ["Tool output summary:"]
        for key, value in list(parsed.items())[:12]:
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    return None


def _looks_like_raw_json_blob(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if text.startswith("{") or text.startswith("["):
        return True
    return '"results"' in text and '"query"' in text and "{" in text


def synthesis_node(state: AgentState) -> dict[str, Any]:
    decision = cast(dict[str, Any], state.get("decision") or {})
    execution_result = cast(dict[str, Any], state.get("execution_result") or {})

    if isinstance(state.get("artifact_result"), dict) and state["artifact_result"].get("success") is True:
        source_material = state["artifact_result"].get("content") or state.get("user_input") or ""
    elif decision.get("action") == "respond":
        source_material = decision.get("final_answer") or state.get("user_input") or ""
    else:
        success = execution_result.get("success")
        content = execution_result.get("content")
        error = execution_result.get("error")
        if success is True:
            structured_summary = _summarize_structured_tool_output(content)
            source_material = structured_summary or (str(content) if content is not None else "")
        else:
            source_material = f"Tool execution failed: {error or 'Unknown error.'}"

    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {"draft_output": str(source_material), "synthesis_draft": str(source_material)}

    try:
        agent = load_agent(agent_key)
        provider = get_provider(agent.provider)
        messages = build_synthesis_messages(
            user_query=str(state.get("user_input", "")),
            raw_data=str(source_material),
            metadata=state.get("metadata"),
            retry_reason=state.get("retry_reason"),
            persona_text=load_agent_persona(agent_key),
        )
        polished = provider.generate(messages)
        polished_text = str(polished or "").strip()
        if not polished_text:
            return {"draft_output": str(source_material), "synthesis_draft": str(source_material)}
        if _looks_like_raw_json_blob(polished_text):
            return {"draft_output": str(source_material), "synthesis_draft": str(source_material)}
        return {"draft_output": polished_text, "synthesis_draft": polished_text}
    except Exception:
        return {"draft_output": str(source_material), "synthesis_draft": str(source_material)}
