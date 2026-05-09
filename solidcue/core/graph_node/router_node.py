from __future__ import annotations

from typing import Any

from solidcue.core.state.schema import AgentState

ARTIFACT_INTENT_KEYWORDS = (
    "resume",
    "document",
    "google doc",
    "pdf",
    "word document",
    "spreadsheet",
    "tracker",
    "save file",
    "upload file",
    "write",
    "generate",
    "create",
)


def _is_artifact_intent(user_input: Any) -> bool:
    if not isinstance(user_input, str):
        return False
    normalized = user_input.casefold()
    return any(keyword in normalized for keyword in ARTIFACT_INTENT_KEYWORDS)


def _retry_limit_reached(state: AgentState) -> bool:
    max_retries = state.get("max_retries") if isinstance(state.get("max_retries"), int) else 10
    total = sum(
        value if isinstance(value, int) else 0
        for value in (
            state.get("source_attempt"),
            state.get("artifact_attempt"),
            state.get("synthesis_attempt"),
        )
    )
    return total >= max_retries


def router_node(state: AgentState) -> dict[str, Any]:
    if _retry_limit_reached(state):
        return {"phase": "final", "failure_type": "retry_limit", "router_next": "final_output"}

    phase = state.get("phase") or "source"
    origin = state.get("router_origin")
    failure_type = state.get("failure_type")

    if origin == "reflection":
        reflection = state.get("reflection_result")
        sufficient = isinstance(reflection, dict) and reflection.get("sufficient") is True
        if not sufficient:
            source_attempt = state.get("source_attempt") if isinstance(state.get("source_attempt"), int) else 0
            return {"phase": "source", "source_attempt": source_attempt + 1, "router_next": "decision"}

        if _is_artifact_intent(state.get("user_input")):
            return {"phase": "artifact", "failure_type": None, "router_next": "decision"}
        return {"phase": "synthesis", "failure_type": None, "router_next": "synthesis"}

    if failure_type is None:
        if phase == "artifact":
            if not isinstance(state.get("artifact_result"), dict):
                return {"phase": "artifact", "router_next": "artifact_generation"}
            return {"phase": "synthesis", "router_next": "synthesis"}
        if phase == "synthesis":
            return {"phase": "final", "router_next": "final_output"}
        return {"phase": "synthesis", "router_next": "synthesis"}

    if failure_type == "missing_source":
        source_attempt = state.get("source_attempt") if isinstance(state.get("source_attempt"), int) else 0
        return {"phase": "source", "source_attempt": source_attempt + 1, "router_next": "decision"}

    if failure_type == "bad_artifact":
        artifact_attempt = state.get("artifact_attempt") if isinstance(state.get("artifact_attempt"), int) else 0
        return {"phase": "artifact", "artifact_attempt": artifact_attempt + 1, "router_next": "artifact_generation"}

    if failure_type == "not_executed":
        artifact_attempt = state.get("artifact_attempt") if isinstance(state.get("artifact_attempt"), int) else 0
        return {"phase": "artifact", "artifact_attempt": artifact_attempt + 1, "router_next": "artifact_execution"}

    if failure_type == "bad_synthesis":
        synthesis_attempt = state.get("synthesis_attempt") if isinstance(state.get("synthesis_attempt"), int) else 0
        return {"phase": "synthesis", "synthesis_attempt": synthesis_attempt + 1, "router_next": "synthesis"}

    return {"phase": "final", "router_next": "final_output"}
