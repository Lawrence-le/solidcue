from __future__ import annotations

import ast
import json
import re
import uuid
import logging
from typing import Any

from solidcue.core.state.schema import AgentState, ToolCallState

from solidcue.agents.configs.loader import load_agent
from solidcue.core.execution.agent_executor import run_agent
from solidcue.tools.loader import load_tool
from solidcue.tools.stages import (
    get_missing_required_tool_fields,
    infer_tool_stage,
    split_missing_tool_fields,
)

logger = logging.getLogger(__name__)

# --- Configuration & Constants ---

TOOL_NAME_ALIASES = {
    "browser_scrape": "scrape_webpage",
    "google_search": "search_web",
    "web_search": "search_web",
    "web_scraper": "scrape_webpage",
    "scrape_url": "scrape_webpage",
    "scrape_web": "scrape_webpage",
    "web_image_search": "search_web_images",
    "web_images_search": "search_web_images",
}

GENERIC_DECISION_FALLBACK = "I couldn't complete that request. Please try again."
ARTIFACT_REQUIRED_RETRY_PREFIX = "ARTIFACT_REQUIRED:"

# --- Hardened Validation Layer ---

class DecisionValidator:
    """Runtime enforcement for the ToolCallState contract."""
    
    @staticmethod
    def validate(
        raw: dict[str, Any],
        available_tools: list[str],
        user_input: str = "",
    ) -> ToolCallState:
        raw = DecisionValidator._canonicalize_raw_decision(raw, user_input)
        action = raw.get("action")
        if action not in ["use_tool", "respond"]:
            action = "respond"

        # --- Case 1: Tool Intent ---
        if action == "use_tool":
            raw_name = raw.get("tool_name")
            tool_name = _resolve_available_tool_name(raw_name, available_tools)
            
            # Fail-Closed: Tool must exist in current configuration
            if not tool_name or tool_name not in available_tools:
                logger.warning(f"Rejecting invalid tool intent: {raw_name}")
                return DecisionValidator.as_fallback_response(
                    "I couldn't safely execute that tool with the current agent configuration. "
                    "Please retry or choose a different request."
                )

            tool_input = raw.get("tool_input") if isinstance(raw.get("tool_input"), dict) else {}
            tool_input = DecisionValidator._enrich_tool_input_from_user_input(
                tool_name,
                tool_input,
                user_input,
            )
            tool_stage = DecisionValidator._resolve_tool_stage(raw.get("tool_stage"), tool_name)
            blocking_missing_fields = DecisionValidator._blocking_missing_required_fields(
                tool_name,
                tool_input,
                tool_stage,
            )
            if blocking_missing_fields:
                fields_text = ", ".join(blocking_missing_fields)
                logger.warning(
                    "Rejecting tool intent with missing required fields for %s: %s",
                    tool_name,
                    fields_text,
                )
                return DecisionValidator.as_fallback_response(
                    "I couldn't safely execute that tool because required inputs were missing "
                    f"({fields_text}). Please retry with complete details."
                )

            return {
                "action": "use_tool",
                "thought": str(raw.get("thought") or "Executing tool..."),
                "tool_stage": tool_stage,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "final_answer": None,
                "approval_preview": raw.get("approval_preview") if isinstance(raw.get("approval_preview"), dict) else None
            }

        # --- Case 2: Response Intent ---
        final_answer = raw.get("final_answer")
        
        return {
            "action": "respond",
            "thought": str(raw.get("thought") or "Responding."),
            "tool_stage": None,
            "tool_name": None,
            "tool_input": {},
            "final_answer": str(final_answer or ""),
            "approval_preview": None
        }

    @staticmethod
    def as_fallback_response(message: str) -> ToolCallState:
        return {
            "action": "respond",
            "thought": "System fallback triggered due to validation failure.",
            "tool_stage": None,
            "tool_name": None,
            "tool_input": {},
            "final_answer": message,
            "approval_preview": None
        }

    @staticmethod
    def _canonicalize_raw_decision(raw: dict[str, Any], user_input: str = "") -> dict[str, Any]:
        tool_call = _extract_openai_style_tool_call(raw)
        if tool_call is None:
            final_answer = raw.get("final_answer")
            if isinstance(final_answer, dict):
                tool_call = _extract_bare_tool_name_intent(final_answer, user_input)
        if tool_call is None:
            return raw

        tool_name, tool_input = tool_call
        return {
            "action": "use_tool",
            "thought": raw.get("thought") or "Recovered tool intent from tool_calls payload.",
            "tool_stage": raw.get("tool_stage"),
            "tool_name": tool_name,
            "tool_input": tool_input,
            "final_answer": None,
            "approval_preview": raw.get("approval_preview"),
        }

    @staticmethod
    def _resolve_tool_stage(raw_stage: Any, tool_name: str) -> str:
        inferred_stage = "context"
        try:
            tool = load_tool(tool_name)
            inferred_stage = infer_tool_stage(tool_name, tool)
        except Exception:
            pass

        if raw_stage in {"context", "artifact"}:
            normalized_stage = str(raw_stage)
            if normalized_stage != inferred_stage:
                logger.info(
                    "Overriding mismatched tool_stage '%s' with inferred stage '%s' for tool '%s'.",
                    normalized_stage,
                    inferred_stage,
                    tool_name,
                )
            return inferred_stage
        return inferred_stage

    @staticmethod
    def _blocking_missing_required_fields(
        tool_name: str,
        tool_input: dict[str, Any],
        tool_stage: str,
    ) -> list[str]:
        try:
            tool = load_tool(tool_name)
        except Exception:
            return []

        missing = get_missing_required_tool_fields(tool, tool_input)
        if tool_stage != "artifact":
            return missing

        _generatable, blocking = split_missing_tool_fields(missing)
        return blocking

    @staticmethod
    def _enrich_tool_input_from_user_input(
        tool_name: str,
        tool_input: dict[str, Any],
        user_input: str,
    ) -> dict[str, Any]:
        enriched = dict(tool_input)

        if "url" not in enriched and _looks_like_url_tool(tool_name):
            url = _extract_first_url(user_input)
            if url:
                enriched["url"] = url

        if "query" not in enriched and _looks_like_search_tool(tool_name):
            query = user_input.strip()
            if query:
                enriched["query"] = query

        return enriched

# --- Primary Node Function ---

def decision_node(state: AgentState) -> dict[str, Any]:
    agent_key = state.get("agent_key")
    user_input = state.get("user_input", "")
    transcript = state.get("messages", [])

    deterministic_source = _select_unread_source_tool(state)
    if deterministic_source is not None:
        tool_name, tool_input, source_name = deterministic_source
        decision = {
            "action": "use_tool",
            "thought": f"Reading source file content before continuing: {source_name}",
            "tool_stage": "context",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "final_answer": None,
            "approval_preview": None,
        }
        return {
            "decision": decision,
            "active_tool_call": decision,
            "tool_use": True,
            "retry_reason": None,
        }

    deterministic_artifact = _select_artifact_tool_for_phase(state)
    if deterministic_artifact is not None:
        tool_name = deterministic_artifact
        decision = {
            "action": "use_tool",
            "thought": "Source gathering is complete; selecting artifact tool.",
            "tool_stage": "artifact",
            "tool_name": tool_name,
            "tool_input": {},
            "final_answer": None,
            "approval_preview": None,
        }
        return {
            "decision": decision,
            "artifact_plan": {"tool_name": tool_name, "tool_input": {}, "thought": decision["thought"]},
            "tool_use": True,
            "phase": "artifact",
            "retry_reason": None,
        }
    
    # 1. Execution
    result = run_agent(
        agent_key=agent_key,
        user_input=user_input,
        transcript=_ensure_user_message(transcript, user_input),
        retry_reason=state.get("retry_reason"),
        metadata=state.get("metadata", {}),
    )

    output_text = result.get("output", "")
    agent_config = result.get("agent_config")
    
    available_tools = []
    if agent_config and hasattr(agent_config, "tools"):
        available_tools = agent_config.tools
    else:
        logger.error(f"Critical: Agent {agent_key} loaded without tool configuration.")

    # 2. Hardened Parsing
    raw_payload = _get_decision_payload(output_text)
    decision = DecisionValidator.validate(raw_payload, available_tools, str(user_input or ""))
    decision = _apply_artifact_retry_override(
        decision,
        available_tools,
        state.get("retry_reason"),
    )
    decision = _apply_neutral_decision_fallback(decision, output_text)
    tool_use = (decision["action"] == "use_tool")

    # 3. Delta Construction
    new_messages: list[dict[str, Any]] = []
    if not any(m.get("role") == "user" for m in transcript):
        new_messages.append({"role": "user", "content": user_input})

    if tool_use and decision.get("tool_stage") != "artifact":
        new_messages.append({
            "role": "assistant",
            "content": "", 
            "tool_calls": [{
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": decision["tool_name"],
                    "arguments": json.dumps(decision["tool_input"]),
                },
            }],
        })
    elif not tool_use:
        new_messages.append({"role": "assistant", "content": decision["final_answer"]})

    # 4. Return Delta for AgentState (LangGraph)
    update = {
        "decision": decision,
        "tool_use": tool_use,
        "messages": new_messages,
        "latest_output": output_text,
        "llm_prompt_messages": result.get("messages", []),
    }

    if tool_use and decision.get("tool_stage") == "context":
        update["active_tool_call"] = decision
    elif tool_use and decision.get("tool_stage") == "artifact":
        update["artifact_plan"] = {
            "tool_name": decision.get("tool_name"),
            "tool_input": decision.get("tool_input") or {},
            "thought": decision.get("thought"),
        }
        update["phase"] = "artifact"

    if not tool_use:
        update["draft_output"] = decision["final_answer"]
        update["finalization_reason"] = "decision_responded"

    return update


def _select_unread_source_tool(state: AgentState) -> tuple[str, dict[str, Any], str] | None:
    manifest = state.get("source_manifest")
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(sources, list):
        return None

    source = next(
        (
            item
            for item in sources
            if isinstance(item, dict)
            and item.get("status") in {"listed", "failed"}
            and isinstance(item.get("id"), str)
            and item.get("id")
        ),
        None,
    )
    if not isinstance(source, dict):
        return None

    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return None

    try:
        agent = load_agent(agent_key)
    except Exception:
        return None

    file_id = str(source.get("id"))
    candidate_tools = sorted(
        [tool for tool in agent.tools or [] if isinstance(tool, str)],
        key=lambda name: (0 if "download" in name.casefold() else 1, name),
    )
    for tool_name in candidate_tools:
        try:
            tool = load_tool(tool_name)
        except Exception:
            continue
        if infer_tool_stage(tool_name, tool) != "context":
            continue
        normalized_name = tool_name.casefold()
        if not any(term in normalized_name for term in ("download", "read", "fetch", "retrieve", "get_file", "get_document", "export")):
            continue
        schema = getattr(getattr(tool, "mcp", None), "input_schema", None)
        properties = schema.get("properties") if isinstance(schema, dict) else None
        property_map = properties if isinstance(properties, dict) else {}
        for field in ("file_id", "document_id", "spreadsheet_id", "id"):
            if field in property_map:
                return tool_name, {field: file_id}, str(source.get("name") or file_id)
    return None


def _select_artifact_tool_for_phase(state: AgentState) -> str | None:
    if state.get("phase") != "artifact":
        return None
    if isinstance(state.get("artifact_plan"), dict) and state["artifact_plan"].get("tool_name"):
        return None
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return None
    try:
        agent = load_agent(agent_key)
    except Exception:
        return None

    artifact_tools: list[str] = []
    for tool_name in agent.tools or []:
        if not isinstance(tool_name, str):
            continue
        try:
            tool = load_tool(tool_name)
            if infer_tool_stage(tool_name, tool) == "artifact":
                artifact_tools.append(tool_name)
        except Exception:
            continue
    preferred = ("docs_create_document", "create_word_document", "create_pdf_document", "create_csv_file")
    for tool_name in preferred:
        if tool_name in artifact_tools:
            return tool_name
    return artifact_tools[0] if artifact_tools else None


def _apply_artifact_retry_override(
    decision: ToolCallState,
    available_tools: list[str],
    retry_reason: Any,
) -> ToolCallState:
    reason = str(retry_reason or "").strip()
    if not reason.startswith(ARTIFACT_REQUIRED_RETRY_PREFIX):
        return decision

    artifact_tools: list[str] = []
    for tool_name in available_tools:
        try:
            tool = load_tool(tool_name)
            if infer_tool_stage(tool_name, tool) == "artifact":
                artifact_tools.append(tool_name)
        except Exception:
            continue

    if not artifact_tools:
        return decision

    if (
        decision.get("action") == "use_tool"
        and decision.get("tool_stage") == "artifact"
        and isinstance(decision.get("tool_name"), str)
        and decision.get("tool_name") in artifact_tools
    ):
        return decision

    selected_tool = _select_preferred_artifact_tool(artifact_tools)
    updated = dict(decision)
    updated["action"] = "use_tool"
    updated["tool_stage"] = "artifact"
    updated["tool_name"] = selected_tool
    updated["tool_input"] = {}
    updated["final_answer"] = None
    updated["thought"] = "Validation requested artifact output; selecting artifact-stage tool."
    return updated


def _select_preferred_artifact_tool(artifact_tools: list[str]) -> str:
    preferred_order = (
        "docs_create_document",
        "create_word_document",
        "create_pdf_document",
        "create_csv_file",
    )
    for tool_name in preferred_order:
        if tool_name in artifact_tools:
            return tool_name
    return artifact_tools[0]


def _apply_neutral_decision_fallback(
    decision: ToolCallState, output_text: str
) -> ToolCallState:
    if decision.get("action") != "respond":
        return decision

    final_answer = str(decision.get("final_answer") or "").strip()
    if final_answer:
        return decision

    message = str(output_text or "").strip() or GENERIC_DECISION_FALLBACK

    updated = dict(decision)
    updated["final_answer"] = message
    return updated


# --- Hierarchical Extraction Utilities ---

def _get_decision_payload(output_text: str) -> dict[str, Any]:
    """Tries extraction methods in order of reliability."""
    # 1. Clean JSON
    json_candidate = _extract_json_candidate(output_text)
    try:
        data = json.loads(json_candidate)
        if isinstance(data, dict) and "action" in data:
            return data
    except Exception:
        pass

    # 2. Legacy/XML Protocols
    parsed = _parse_tool_call_output(output_text) or _parse_xml_tool_call_output(output_text)
    if parsed:
        return parsed
    
    # 3. Balanced Object (Broken JSON salvage)
    balanced = _extract_balanced_object_candidate(output_text)
    if balanced:
        try:
            data = json.loads(balanced)
            if isinstance(data, dict): return data
        except Exception:
            pass

    salvaged = _salvage_tool_intent(output_text)
    if salvaged:
        return salvaged

    return {"action": "respond", "final_answer": output_text}


def _extract_openai_style_tool_call(raw: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    container: Any = raw
    final_answer = raw.get("final_answer")
    if isinstance(final_answer, dict):
        container = final_answer

    if not isinstance(container, dict):
        return None

    tool_calls = container.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None

    tool_name = first_call.get("tool_name") or first_call.get("function") or first_call.get("name")
    tool_input = first_call.get("tool_args") or first_call.get("args") or first_call.get("arguments")

    function_call = first_call.get("function")
    if isinstance(function_call, dict):
        tool_name = tool_name or function_call.get("name")
        arguments = function_call.get("arguments")
        if isinstance(arguments, dict):
            tool_input = tool_input or arguments
        elif isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = None
            if isinstance(parsed_arguments, dict):
                tool_input = tool_input or parsed_arguments

    if not isinstance(tool_name, str) or not tool_name.strip():
        return None

    normalized_input = tool_input if isinstance(tool_input, dict) else {}
    return tool_name.strip(), normalized_input


def _extract_bare_tool_name_intent(
    payload: dict[str, Any],
    user_input: str,
) -> tuple[str, dict[str, Any]] | None:
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None

    tool_input = payload.get("tool_input") or payload.get("tool_args") or payload.get("args")
    normalized_input = tool_input if isinstance(tool_input, dict) else {}

    if "url" not in normalized_input:
        url = _extract_first_url(user_input)
        if url and _looks_like_url_tool(tool_name):
            normalized_input = {**normalized_input, "url": url}

    if "query" not in normalized_input and _looks_like_search_tool(tool_name):
        query = user_input.strip()
        if query:
            normalized_input = {**normalized_input, "query": query}

    return tool_name.strip(), normalized_input


def _extract_first_url(text: str) -> str | None:
    match = re.search(r"https?://[^\s)>\]]+", text)
    if not match:
        return None
    return match.group(0).rstrip(".,")


def _looks_like_url_tool(tool_name: str) -> bool:
    normalized = tool_name.casefold()
    return any(term in normalized for term in ("browser", "scrape", "url", "webpage"))


def _looks_like_search_tool(tool_name: str) -> bool:
    normalized = tool_name.casefold()
    return "search" in normalized or "google" in normalized


def _looks_like_tool_intent_json(text: str) -> bool:
    lowered = text.casefold()
    return '"action":"use_tool"' in lowered and '"tool_name"' in lowered


def _salvage_tool_intent(text: str) -> dict[str, Any] | None:
    if not _looks_like_tool_intent_json(text):
        return None

    tool_name_match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', text)
    if not tool_name_match:
        return None

    tool_name = tool_name_match.group(1).strip()
    if not tool_name:
        return None

    tool_input: dict[str, Any] = {}
    tool_input_match = re.search(r'"tool_input"\s*:\s*(\{.*\})', text, flags=re.DOTALL)
    if tool_input_match:
        candidate = tool_input_match.group(1)
        candidate = re.sub(r',\s*"(?:final_answer|approval_preview)"\s*:\s*.*$', "", candidate, flags=re.DOTALL)
        opens = candidate.count("{")
        closes = candidate.count("}")
        if closes < opens:
            candidate = f"{candidate}{'}' * (opens - closes)}"
        try:
            parsed_input = json.loads(candidate)
            if isinstance(parsed_input, dict):
                tool_input = parsed_input
        except json.JSONDecodeError:
            pass

    return {
        "action": "use_tool",
        "thought": "Recovered tool intent from malformed decision payload.",
        "tool_stage": None,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "final_answer": None,
        "approval_preview": None,
    }

def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
    if stripped.startswith("{"):
        return stripped

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "action" in candidate:
            return json.dumps(candidate)
    return stripped

def _extract_balanced_object_candidate(text: str) -> str | None:
    start = text.find("{")
    if start == -1: return None
    depth, in_string, escaped = 0, False, False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped: escaped = False
            elif ch == "\\": escaped = True
            elif ch == '"': in_string = False
            continue
        if ch == '"': in_string = True
        elif ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return text[start : idx + 1]
    return None

def _parse_tool_call_output(text: str) -> dict[str, Any] | None:
    match = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", text, re.DOTALL)
    if not match:
        return None

    payload = match.group(1).strip()
    if payload.startswith("[") and payload.endswith("]"):
        payload = payload[1:-1].strip()

    try:
        expr = ast.parse(payload, mode="eval").body
    except SyntaxError:
        return None

    if not isinstance(expr, ast.Call):
        return None

    tool_name: str | None = None
    if isinstance(expr.func, ast.Name):
        tool_name = expr.func.id
    elif isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
        tool_name = expr.func.attr

    if not tool_name:
        return None

    kwargs: dict[str, Any] = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            continue
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            return None

    explicit_tool_input = kwargs.get("tool_input")
    if isinstance(explicit_tool_input, dict):
        tool_input = explicit_tool_input
    else:
        tool_input = {k: v for k, v in kwargs.items() if k != "tool_name"}

    return {
        "action": "use_tool",
        "thought": None,
        "tool_stage": None,
        "tool_name": kwargs.get("tool_name") or tool_name,
        "tool_input": tool_input,
        "final_answer": None,
        "approval_preview": None,
    }

def _parse_xml_tool_call_output(text: str) -> dict[str, Any] | None:
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    payload = match.group(1).strip()
    if not payload:
        return None

    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if not lines:
        return None

    tool_name = lines[0]
    if not tool_name:
        return None

    keys = re.findall(r"<arg_key>\s*(.*?)\s*</arg_key>", payload, flags=re.DOTALL | re.IGNORECASE)
    values = re.findall(r"<arg_value>\s*(.*?)\s*</arg_value>", payload, flags=re.DOTALL | re.IGNORECASE)

    tool_input: dict[str, Any] = {}
    for key, value in zip(keys, values):
        parsed_value: Any = value.strip()
        if isinstance(parsed_value, str):
            lower = parsed_value.lower()
            if lower == "null":
                parsed_value = None
            elif lower == "true":
                parsed_value = True
            elif lower == "false":
                parsed_value = False
            else:
                try:
                    parsed_value = json.loads(parsed_value)
                except Exception:
                    pass
        tool_input[key.strip()] = parsed_value

    return {
        "action": "use_tool",
        "thought": None,
        "tool_stage": None,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "final_answer": None,
        "approval_preview": None,
    }

# --- State & Alias Helpers ---

def _resolve_available_tool_name(tool_name: Any, available_tools: list[str]) -> str | None:
    if not isinstance(tool_name, str): return None
    if tool_name in available_tools: return tool_name
    return TOOL_NAME_ALIASES.get(tool_name)

def _ensure_user_message(transcript: list[dict[str, Any]], user_input: str) -> list[dict[str, Any]]:
    if any(m.get("role") == "user" for m in transcript):
        return transcript.copy()
    return [{"role": "user", "content": user_input}] + transcript
