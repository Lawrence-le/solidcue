from __future__ import annotations

import asyncio

from langgraph.runtime import Runtime

from solidcue.agent_configs.loader import list_agents
from solidcue.core.graph_router.nodes._shared import (
    _PROFILE_ROUTER_PROVIDER,
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


def _normalize_plan(
    raw_plan: object,
    *,
    user_input: str,
    valid_agent_keys: set[str],
) -> list[dict[str, str]]:
    """Coerce the model's plan into a validated list of {agent_key, sub_task} steps.

    Unknown agent keys are dropped. Steps missing a sub_task fall back to the
    original user input so the worker still receives a task.
    """
    plan: list[dict[str, str]] = []
    if isinstance(raw_plan, list):
        for step in raw_plan:
            if not isinstance(step, dict):
                continue
            agent_key = normalize_text(step.get("agent_key"))
            if not agent_key or agent_key not in valid_agent_keys:
                continue
            sub_task = normalize_text(step.get("sub_task")) or user_input
            plan.append({"agent_key": agent_key, "sub_task": sub_task})
    return plan


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
        # Under LangGraph Server there is no FastAPI layer to seed the in-memory
        # cache, so fall back to the user's profile provider loaded at import time.
        provider = _PROFILE_ROUTER_PROVIDER
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
        metadata=state.get("metadata"),
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
    route_reason = normalize_text(parsed.get("route_reason"))

    valid_agent_keys = {agent["agent_key"] for agent in available_agents}
    plan = _normalize_plan(
        parsed.get("plan"),
        user_input=user_input,
        valid_agent_keys=valid_agent_keys,
    )

    if router_intent == "task" and not plan:
        # The model classified this as work but gave no usable plan. Honor an
        # explicit (legacy) target_agent_key if present, otherwise fall back to
        # the single-agent heuristic so the request still gets routed.
        legacy_key = normalize_text(parsed.get("target_agent_key"))
        fallback_key = legacy_key if legacy_key in valid_agent_keys else select_target_agent_key(user_input)
        if fallback_key in valid_agent_keys:
            plan = [{"agent_key": fallback_key, "sub_task": user_input}]

    if router_intent == "task" and not plan:
        # No agent can take this work — ask for clarification instead of dropping it.
        message = "I don't have an agent that can handle that yet. Could you clarify what you need?"
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "route_reason": route_reason or "No matching agent for the requested task.",
            "target_agent_key": "",
            "assistant_draft": message,
            "final_response": message,
            "plan": [],
            "handoff": {},
        }

    # Keep the single-agent fields populated for backward compatibility: the first
    # step is the primary handoff target.
    target_agent_key = plan[0]["agent_key"] if plan else ""

    if plan and router_intent == "task":
        router_next = "handoff"

    handoff = parsed.get("handoff")
    if not isinstance(handoff, dict) and router_next == "handoff":
        handoff = {
            "action": "create_agent" if router_intent == "create_agent" else "route_agent",
            "task_input": user_input,
            "target_agent_key": target_agent_key,
        }

    raw_sources = parsed.get("target_artifacts_source")
    target_artifacts_source = raw_sources if isinstance(raw_sources, list) else []

    result: dict[str, object] = {
        "router_intent": router_intent,
        "router_next": router_next,
        "route_reason": route_reason,
        "target_agent_key": target_agent_key,
        "assistant_draft": assistant_draft,
        "final_response": assistant_draft,
        "plan": plan,
        "handoff": handoff if isinstance(handoff, dict) else {},
        "target_artifacts_source": target_artifacts_source,
    }

    # create_agent: once the model signals it has gathered a name + purpose, seed
    # the system graph so the embedded subgraph runs the actual build. Until then
    # the router just keeps conversing.
    if router_intent == "create_agent":
        raw_spec = parsed.get("agent_spec")
        agent_ready = bool(parsed.get("agent_ready")) and isinstance(raw_spec, dict)
        result["assistant_draft"] = _build_create_agent_reply(agent_ready=agent_ready)
        result["final_response"] = result["assistant_draft"]
        if agent_ready:
            name = normalize_text(raw_spec.get("name"))
            agent_key = normalize_text(raw_spec.get("agent_key"))
            description = normalize_text(raw_spec.get("description"))
            # Tool selection is handled by graph_system's select_tools node — the
            # router only gathers name/purpose.
            if name and agent_key and description:
                result["agent_spec"] = {
                    "name": name,
                    "agent_key": agent_key,
                    "description": description,
                }
                result["system_intent"] = "create_agent"

    return result
