from __future__ import annotations

import asyncio
from typing import Any

from langgraph.runtime import Runtime

from solidcue.core.graph_router.nodes._shared import (
    available_agents,
    extract_json_object,
    normalize_text,
    resolve_router_provider,
)
from solidcue.core.graph_router.prompts.router_prompt import (
    build_router_messages,
)
from solidcue.core.graph_router.state.schema import RouterState
from solidcue.core.utils.generation import (
    generate_full_then_parse,
)


_JSON_ONLY_REMINDER = (
    "Your previous reply was not valid JSON. Reply again with ONLY a single JSON object "
    "matching the required shape. Keep assistant_draft to one short sentence with no "
    "tables or line breaks; escape any newline as \\n."
)


def _build_create_agent_reply(*, agent_ready: bool = False) -> str:
    if agent_ready:
        return (
            "**Agent details received**\n\n"
            "I'm ready to create it now."
        )
    return (
        "**Yes, I can help with that**\n\n"
        "What I need:\n"
        "- What should the agent do?\n"
        "- What would you like to call it?\n"
    )


def _clarify(message: str, *, reason: str) -> dict[str, object]:
    return {
        "router_intent": "clarify",
        "router_next": "final_output",
        "route_reason": reason,
        "assistant_draft": message,
        "final_response": message,
    }


async def intent_router_node(
    state: RouterState,
    *,
    runtime: Runtime[RouterState] | None = None,
) -> dict[str, object]:
    """Classify the user's intent only.

    This node decides *what kind* of request this is (chat, task, reshape,
    create_agent, clarify). It does NOT build the execution plan — that is the job
    of build_plan_node, which runs only for the task intent. Keeping classification
    separate keeps simple turns (chat/reshape) fast: they never pay for plan writing.
    """
    user_input = normalize_text(state.get("user_input"))
    if not user_input:
        return {
            "router_intent": "chat",
            "router_next": "final_output",
            "assistant_draft": "",
            "final_response": "",
            "route_reason": "",
        }

    thread_id = normalize_text(state.get("thread_id"))
    try:
        provider = resolve_router_provider(thread_id)
    except ValueError as exc:
        message = normalize_text(str(exc)) or "Router provider configuration is invalid."
        return _clarify(message, reason="Router provider configuration is invalid.")
    if provider is None:
        return _clarify(
            "Select a provider and model for the router first.",
            reason="Router provider configuration is missing.",
        )

    agents = available_agents()
    messages = build_router_messages(
        user_input=user_input,
        chat_history=state.get("chat_history"),
        available_agents=agents,
        metadata=state.get("metadata"),
        agent_results=state.get("agent_results"),
    )

    # Small router models occasionally return prose or malformed JSON (e.g. a real
    # newline inside a string). Retry once with a strict reminder before giving up, so a
    # single bad generation doesn't dead-end the turn.
    parsed: Any = None
    attempts = [messages, messages + [{"role": "user", "content": _JSON_ONLY_REMINDER}]]
    for attempt_messages in attempts:
        try:
            # Blocking sync HTTP call — run in a worker thread so it never freezes the loop.
            parsed, _routing_metric, _routing_output = await asyncio.to_thread(
                generate_full_then_parse,
                provider,
                attempt_messages,
                extract_json_object,
                node_name="intent_router",
            )
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            break

    if not isinstance(parsed, dict):
        # Still no parseable response — degrade to a friendly rephrase prompt rather
        # than an internal-error message, so the user has a clear next step.
        return _clarify(
            "Sorry, I didn't quite catch that — could you rephrase your request?",
            reason="Router model did not return valid JSON after retry.",
        )

    assistant_draft = normalize_text(parsed.get("assistant_draft")) or "I can help with that."
    router_intent = normalize_text(parsed.get("router_intent")) or normalize_text(parsed.get("intent"))
    route_reason = normalize_text(parsed.get("route_reason"))

    result: dict[str, object] = {
        "router_intent": router_intent,
        "router_next": "final_output",
        "route_reason": route_reason,
        "assistant_draft": assistant_draft,
        "final_response": assistant_draft,
    }

    # create_agent: gather name + purpose conversationally, then seed the system graph.
    # This is spec collection, not an execution plan, so it stays in the router.
    if router_intent == "create_agent":
        raw_spec = parsed.get("agent_spec")
        agent_ready = bool(parsed.get("agent_ready")) and isinstance(raw_spec, dict)
        result["assistant_draft"] = _build_create_agent_reply(agent_ready=agent_ready)
        result["final_response"] = result["assistant_draft"]
        if agent_ready:
            name = normalize_text(raw_spec.get("name"))
            agent_key = normalize_text(raw_spec.get("agent_key"))
            description = normalize_text(raw_spec.get("description"))
            if name and agent_key and description:
                result["agent_spec"] = {
                    "name": name,
                    "agent_key": agent_key,
                    "description": description,
                }
                result["system_intent"] = "create_agent"

    return result
