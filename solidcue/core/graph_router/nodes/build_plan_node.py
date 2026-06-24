"""
build_plan_node: writes the execution plan for a task-intent turn.

Decoupled from intent_router_node so classification stays cheap: simple turns
(chat/reshape/clarify) never reach this node and never pay for plan generation. This
node runs only after the router classifies a turn as `task`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from solidcue.core.graph_router.nodes._shared import (
    available_agents,
    extract_json_object,
    normalize_text,
    resolve_router_provider,
    select_target_agent_key,
)
from solidcue.core.graph_router.prompts.router_plan_prompt import build_plan_messages
from solidcue.core.graph_router.state.schema import RouterState
from solidcue.core.utils.generation import generate_full_then_parse


def _normalize_plan(
    raw_plan: object,
    *,
    user_input: str,
    valid_agent_keys: set[str],
) -> list[dict[str, str]]:
    """Coerce the model's plan into validated {agent_key, sub_task} steps.

    Unknown agent keys are dropped; steps missing a sub_task fall back to the user input.
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


def _no_plan_clarify(reason: str) -> dict[str, Any]:
    message = "I don't have an agent that can handle that yet. Could you clarify what you need?"
    return {
        "router_intent": "clarify",
        "router_next": "final_output",
        "route_reason": reason,
        "assistant_draft": message,
        "final_response": message,
        "plan": [],
        "handoff": {},
    }


async def build_plan_node(state: RouterState) -> dict[str, Any]:
    user_input = normalize_text(state.get("user_input"))
    thread_id = normalize_text(state.get("thread_id"))

    try:
        provider = resolve_router_provider(thread_id)
    except ValueError:
        provider = None

    agents = available_agents()
    valid_agent_keys = {agent["agent_key"] for agent in agents}

    plan: list[dict[str, str]] = []
    target_artifacts_source: list[Any] = []

    if provider is not None:
        messages = build_plan_messages(
            user_input=user_input,
            available_agents=agents,
            chat_history=state.get("chat_history"),
            agent_results=state.get("agent_results"),
            metadata=state.get("metadata"),
        )
        try:
            parsed, _metric, _output = await asyncio.to_thread(
                generate_full_then_parse,
                provider,
                messages,
                extract_json_object,
                node_name="build_plan",
            )
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            plan = _normalize_plan(
                parsed.get("plan"), user_input=user_input, valid_agent_keys=valid_agent_keys
            )
            raw_sources = parsed.get("target_artifacts_source")
            target_artifacts_source = raw_sources if isinstance(raw_sources, list) else []

    # Fallback: model gave no usable plan but this is task work — route to a single
    # agent via the keyword heuristic so the request is still handled.
    if not plan:
        fallback_key = select_target_agent_key(user_input)
        if fallback_key in valid_agent_keys:
            plan = [{"agent_key": fallback_key, "sub_task": user_input}]

    if not plan:
        return _no_plan_clarify("No matching agent for the requested task.")

    target_agent_key = plan[0]["agent_key"]
    return {
        "router_intent": "task",
        "router_next": "handoff",
        "plan": plan,
        "target_agent_key": target_agent_key,
        "target_artifacts_source": target_artifacts_source,
        "handoff": {
            "action": "route_agent",
            "task_input": user_input,
            "target_agent_key": target_agent_key,
        },
    }
