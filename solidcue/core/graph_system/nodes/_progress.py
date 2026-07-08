"""Stream the create-agent build as sub-agent progress events.

Reuses the existing frontend panel (`plan` + `subagent` events, same shape as
`execute_plan_node`) so agent creation shows a dispatch dialog with no frontend
change. Emission is best-effort: outside a streaming run (e.g. unit tests)
`get_stream_writer()` has no writer, so every call is a no-op.
"""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

# Build steps shown as cards in the reused sub-agent panel, in execution order.
BUILD_STEPS = ("Select tools", "Write definition files", "Write config", "Verify")


def _writer() -> Any:
    try:
        return get_stream_writer()
    except Exception:
        return None


def emit_plan(agent_key: str) -> None:
    """Announce the build steps up front so the panel renders all cards as pending."""
    writer = _writer()
    if not writer:
        return
    try:
        writer(
            {
                "event": "plan",
                "data": {
                    "intro": f"Creating agent '{agent_key}'",
                    "route_reason": "",
                    "steps": [
                        {"agent_key": agent_key, "sub_task": label, "step_index": idx}
                        for idx, label in enumerate(BUILD_STEPS)
                    ],
                    "step_count": len(BUILD_STEPS),
                },
            }
        )
    except Exception:
        pass


def emit_step(agent_key: str, step_index: int, status: str) -> None:
    """Update one step's status: 'running' | 'completed' | 'failed'."""
    writer = _writer()
    if not writer or not 0 <= step_index < len(BUILD_STEPS):
        return
    try:
        writer(
            {
                "event": "subagent",
                "data": {
                    "agent_key": agent_key,
                    "sub_task": BUILD_STEPS[step_index],
                    "step_index": step_index,
                    "step_count": len(BUILD_STEPS),
                    "status": status,
                },
            }
        )
    except Exception:
        pass
