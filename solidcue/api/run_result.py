"""Helpers that shape a raw LangGraph run result into an HTTP response.

This is transport-layer presentation only — it reads fields off the result the
service returns and decides whether the run is paused (interrupt) or done. It
adds no orchestration logic; the equivalent presentation for the CLI lives in
``solidcue.app.agent.commands``.
"""

from __future__ import annotations

from typing import Any

from solidcue.api.schemas import RunAgentResponse

_OUTPUT_KEYS = ("final_output", "final_response", "synthesis_draft", "draft_output")


def extract_interrupt_payload(result: Any) -> dict[str, Any] | None:
    """Return the first interrupt's value dict, or None if the run is not paused."""
    # v1 dict-style interrupt envelope
    if isinstance(result, dict):
        raw_interrupts = result.get("__interrupt__")
    else:
        # v2 GraphOutput style
        raw_interrupts = getattr(result, "interrupts", None)

    if isinstance(raw_interrupts, (list, tuple)) and raw_interrupts:
        value = getattr(raw_interrupts[0], "value", None)
        if isinstance(value, dict):
            return value
    return None


def first_output(result: dict[str, Any]) -> str | None:
    for key in _OUTPUT_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_run_response(*, thread_id: str, result: Any) -> RunAgentResponse:
    interrupt = extract_interrupt_payload(result)
    if interrupt is not None:
        return RunAgentResponse(
            thread_id=thread_id,
            status="interrupted",
            interrupt=interrupt,
        )

    state = result if isinstance(result, dict) else {}
    phase = state.get("phase")
    current_task = state.get("current_task")
    return RunAgentResponse(
        thread_id=thread_id,
        status="completed",
        output=first_output(state) or "No final response generated.",
        phase=phase if isinstance(phase, str) else None,
        current_task=current_task if isinstance(current_task, str) else None,
    )
