from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from solidcue.agents.configs.loader import list_agents
from solidcue.core.router_graph.router_node._shared import (
    extract_json_object,
    get_runtime_router_provider,
    normalize_text,
    select_target_agent_key,
)
from solidcue.core.router_graph.state import RouterState
from solidcue.core.utils.metrics import timed_generate
from solidcue.prompts.router_prompt import build_router_messages


def _is_after_task(state: RouterState, user_input: str) -> bool:
    chat_history = state.get("chat_history")
    if isinstance(chat_history, list) and len(chat_history) > 1:
        return True

    lowered = user_input.casefold()
    if any(
        keyword in lowered
        for keyword in ("follow up", "follow-up", "continue", "resume", "refine", "change")
    ):
        return True

    return False


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


def _looks_like_capability_question(user_input: str) -> bool:
    lowered = user_input.casefold().strip()
    if "?" not in lowered:
        return False

    question_prefixes = (
        "can you ",
        "could you ",
        "would you ",
        "are you able to ",
        "do you think you can ",
        "will you ",
    )
    if not lowered.startswith(question_prefixes):
        return False

    task_verbs = (
        "generate",
        "create",
        "build",
        "write",
        "update",
        "edit",
        "archive",
        "research",
        "analyze",
        "make",
        "prepare",
    )
    return any(f" {verb} " in lowered for verb in task_verbs)


def _capability_question_response(user_input: str) -> dict[str, object]:
    lowered = user_input.casefold()
    if "resume" in lowered:
        question = "Yes. Do you want me to generate the resume now?"
    elif "job" in lowered or "jd" in lowered:
        question = "Yes. Do you want me to run that task now?"
    else:
        question = "Yes. Do you want me to proceed with that task now?"

    return {
        "router_intent": "clarify",
        "router_next": "final_output",
        "route_reason": "User asked whether the task can be done before explicitly requesting execution.",
        "assistant_draft": question,
        "final_response": question,
        "target_agent_key": "",
    }


def _fallback_route(state: RouterState, user_input: str) -> dict[str, object]:
    lowered = user_input.casefold()
    if _looks_like_capability_question(user_input):
        return _capability_question_response(user_input)

    if "create agent" in lowered or "new agent" in lowered:
        return {
            "router_intent": "create_agent",
            "router_next": "handoff",
            "route_reason": "User asked to create a new agent.",
            "target_agent_key": "",
            "handoff": {
                "action": "create_agent",
                "task_input": user_input,
            },
        }

    if any(keyword in lowered for keyword in ("hello", "hi", "thanks", "thank you")):
        return {
            "router_intent": "chat",
            "router_next": "final_output",
            "assistant_draft": "How can I help?",
            "final_response": "How can I help?",
            "target_agent_key": "",
        }

    if _is_after_task(state, user_input) and not state.get("chat_history"):
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "route_reason": "Need more context from prior work before routing.",
            "assistant_draft": "Can you add a little more detail?",
            "final_response": "Can you add a little more detail?",
            "target_agent_key": "",
        }

    return {
        "router_intent": "task",
        "router_next": "handoff",
        "route_reason": "User request should be handled by an agent graph.",
        "target_agent_key": select_target_agent_key(user_input),
        "handoff": {
            "action": "route_agent",
            "task_input": user_input,
        },
    }


def intent_router_node(
    state: RouterState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    user_input = normalize_text(state.get("user_input"))
    if not user_input:
        return {
            "router_intent": "chat",
            "router_next": "final_output",
            "assistant_draft": "",
            "final_response": "",
        }

    if _looks_like_capability_question(user_input):
        return _capability_question_response(user_input)

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
        }
    if provider is None:
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "route_reason": "Router provider configuration is missing.",
            "assistant_draft": "Select a provider and model for the router first.",
            "final_response": "Select a provider and model for the router first.",
            "target_agent_key": "",
        }

    available_agents = _available_agents()
    messages = build_router_messages(
        user_input=user_input,
        chat_history=state.get("chat_history"),
        available_agents=available_agents,
    )

    try:
        response_text, _metric_stats = timed_generate(
            provider,
            messages,
            node_name="router_intent",
        )
    except Exception:
        return _fallback_route(state, user_input)

    parsed = extract_json_object(str(response_text or ""))
    if not isinstance(parsed, dict):
        return _fallback_route(state, user_input)

    intent = normalize_text(parsed.get("intent"))
    response = normalize_text(parsed.get("response"))
    route_reason = normalize_text(parsed.get("route_reason"))
    target_agent_key = normalize_text(parsed.get("target_agent_key"))
    valid_agent_keys = {agent["agent_key"] for agent in available_agents}
    if target_agent_key and target_agent_key not in valid_agent_keys:
        target_agent_key = ""

    if intent == "create_agent":
        return {
            "router_intent": "create_agent",
            "router_next": "handoff",
            "route_reason": route_reason or "User asked to create a new agent.",
            "target_agent_key": "",
            "assistant_draft": response,
            "handoff": {
                "action": "create_agent",
                "task_input": user_input,
            },
        }

    if intent == "chat":
        final_response = response or "How can I help?"
        return {
            "router_intent": "chat",
            "router_next": "final_output",
            "assistant_draft": final_response,
            "final_response": final_response,
            "route_reason": route_reason or "Direct chat response.",
            "target_agent_key": "",
        }

    if intent == "clarify":
        final_response = response or "Can you add a little more detail?"
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "assistant_draft": final_response,
            "final_response": final_response,
            "route_reason": route_reason or "Need clarification before routing.",
            "target_agent_key": "",
        }

    if _looks_like_capability_question(user_input):
        return _capability_question_response(user_input)

    selected_agent_key = target_agent_key or select_target_agent_key(user_input)
    if not selected_agent_key:
        return {
            "router_intent": "clarify",
            "router_next": "final_output",
            "assistant_draft": response or "I need a little more context before I can route this.",
            "final_response": response or "I need a little more context before I can route this.",
            "route_reason": route_reason or "No target agent matched the request.",
            "target_agent_key": "",
        }
    return {
        "router_intent": "task",
        "router_next": "handoff",
        "route_reason": route_reason or "User request should be handled by an agent graph.",
        "target_agent_key": selected_agent_key,
        "handoff": {
            "action": "route_agent",
            "task_input": user_input,
            "target_agent_key": selected_agent_key,
        },
    }
