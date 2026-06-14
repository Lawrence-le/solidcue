from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from solidcue.agent_configs.loader import list_agents
from solidcue.core.graph_router.nodes._shared import (
    extract_json_object,
    get_runtime_router_provider,
    normalize_text,
    select_target_agent_key,
)
from solidcue.core.graph_router.prompts.router_prompt import (
    build_router_messages,
)
from solidcue.core.graph_router.state.schema import RouterState
from solidcue.core.utils.generation import (
    generate_full_then_parse,
)


def _available_agents() -> list[dict[str, str]]:
    agents: list[dict[str, str]] = []
    for agent in list_agents():
        agent_key = normalize_text(getattr(agent, "agent_key", ""))
        if not agent_key:
            continue
        agents.append(
            {
                "agent_key": agent_key,
                "name": normalize_text(getattr(agent, "name", "")),
                "description": normalize_text(getattr(agent, "description", "")),
            }
        )
    return agents


async def intent_router_node(
    state: RouterState,
    *,
    runtime: Runtime[RouterState] | None = None,
) -> dict[str, object]:
    user_input = normalize_text(state.get("user_input"))
    if not user_input:
        return {
            "router_intent": "chat",
            "router_next": "final_output",
            "assistant_draft": "",
            "final_response": "",
            "target_agent_key": "",
            "route_reason": "",
            "handoff": {},
        }

    thread_id = normalize_text(state.get("thread_id"))
    try:
        provider = get_runtime_router_provider(thread_id)
    except ValueError as exc:
        message = normalize_text(str(exc)) or "Router provider configuration is invalid."
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "route_reason": "Router provider configuration is invalid.",
            "assistant_draft": message,
            "final_response": message,
            "target_agent_key": "",
            "handoff": {},
        }
    if provider is None:
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "route_reason": "Router provider configuration is missing.",
            "assistant_draft": "Select a provider and model for the router first.",
            "final_response": "Select a provider and model for the router first.",
            "target_agent_key": "",
            "handoff": {},
        }

    available_agents = _available_agents()
    messages = build_router_messages(
        user_input=user_input,
        chat_history=state.get("chat_history"),
        available_agents=available_agents,
    )

    try:
        # This node runs as a coroutine, so it is awaited directly on the event
        # loop. generate_full_then_parse drives a blocking sync HTTP call, which
        # would freeze the loop for the whole router generation and stall every
        # other request — including session-refresh reads — so run it in a
        # worker thread.
        parsed, _routing_metric, _routing_output = await asyncio.to_thread(
            generate_full_then_parse,
            provider,
            messages,
            extract_json_object,
            node_name="intent_router",
        )
    except Exception:
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "route_reason": "Router model generation failed.",
            "assistant_draft": "I couldn't generate a router response.",
            "final_response": "I couldn't generate a router response.",
            "target_agent_key": "",
            "handoff": {},
        }

    if not isinstance(parsed, dict):
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "route_reason": "Router model did not return valid JSON.",
            "assistant_draft": "I couldn't generate a router response.",
            "final_response": "I couldn't generate a router response.",
            "target_agent_key": "",
            "handoff": {},
        }

    assistant_draft = normalize_text(parsed.get("assistant_draft"))
    if not assistant_draft:
        assistant_draft = "I can help with that."
    router_intent = normalize_text(parsed.get("router_intent")) or normalize_text(
        parsed.get("intent")
    )
    router_next = normalize_text(parsed.get("router_next"))
    target_agent_key = normalize_text(parsed.get("target_agent_key"))
    if not target_agent_key and router_intent == "task":
        target_agent_key = select_target_agent_key(user_input)
    route_reason = normalize_text(parsed.get("route_reason"))
    handoff = parsed.get("handoff")
    if not isinstance(handoff, dict) and router_next == "handoff":
        handoff = {
            "action": "create_agent" if router_intent == "create_agent" else "route_agent",
            "task_input": user_input,
            "target_agent_key": target_agent_key,
        }

    return {
        "router_intent": router_intent,
        "router_next": router_next,
        "route_reason": route_reason,
        "target_agent_key": target_agent_key,
        "assistant_draft": assistant_draft,
        "final_response": assistant_draft,
        "handoff": handoff if isinstance(handoff, dict) else {},
    }
