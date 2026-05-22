import json
import logging
from typing import Any

from solidcue.agents.configs.loader import load_agent, load_agent_persona
from solidcue.core.execution.provider_resolver import get_provider_for_role
from solidcue.core.state.schema import AgentState
from solidcue.core.utils.metrics import build_metric_state_delta, timed_generate
from solidcue.prompts.classifier_prompt import build_classifier_messages

logger = logging.getLogger(__name__)

"""
Classifier Node - Function Overview
-----------------------------------

_extract_json_object:
Parse classifier LLM output into a JSON dict safely.

classifier_node:
Main entrypoint. Phases:
1) Validate input/agent context
2) Run intent classifier prompt
3) Route to conversational flow or planning flow
"""

# ---------------------------------------------------------------------------
# Section: parser helper
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Section: core node
# ---------------------------------------------------------------------------

def classifier_node(state: AgentState) -> dict[str, Any]:
    # Phase 1: validate required state.
    agent_key = state.get("agent_key")
    if not isinstance(agent_key, str) or not agent_key:
        return {}

    user_input = state.get("user_input", "")
    if not user_input.strip():
        return {"phase": "conversational", "router_next": "final_output", "final_response": ""}

    agent = load_agent(agent_key)
    # Phase 2: build classifier prompt and infer intent.
    messages = build_classifier_messages(
        user_input=user_input,
        persona=load_agent_persona(agent_key),
        agent_name=agent.name or "",
        agent_description=agent.description or "",
    )

    provider = get_provider_for_role(agent, "lite")
    response_text, metric_stats = timed_generate(provider, messages)

    parsed = _extract_json_object(str(response_text or ""))
    intent = parsed.get("intent") if isinstance(parsed, dict) else None

    # Phase 3: route by classified intent.
    if intent == "greeting":
        return {
            "phase": "conversational",
            "router_next": "final_output",
            "final_response": str(parsed.get("response", "")),
            **build_metric_state_delta("classifier", "metric_classifier", metric_stats),
            "messages": [{"role": "system", "content": "Greeting — skipping task pipeline"}],
        }

    if intent == "off_topic":
        return {
            "phase": "conversational",
            "router_next": "final_output",
            "final_response": str(parsed.get("response", "")),
            **build_metric_state_delta("classifier", "metric_classifier", metric_stats),
            "messages": [{"role": "system", "content": "Off-topic query — skipping task pipeline"}],
        }

    if intent == "conversational":
        return {
            "phase": "conversational",
            "router_next": "planning",
            **build_metric_state_delta("classifier", "metric_classifier", metric_stats),
            "messages": [{"role": "system", "content": "Conversational question — routing to planning"}],
        }

    return {
        **build_metric_state_delta("classifier", "metric_classifier", metric_stats),
        "messages": [{"role": "system", "content": "Task intent detected — proceeding to discovery"}],
    }
