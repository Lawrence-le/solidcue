from __future__ import annotations

from typing import Any

from solidcue.agent_configs.loader import (
    get_persona_path,
    get_skill_path,
    get_tools_path,
    load_agent,
)
from solidcue.core.graph_system.state.schema import SystemState


def verify_node(state: SystemState) -> dict[str, Any]:
    """Confirm the agent folder is complete: YAML loads and all three MD files exist."""
    from solidcue.core.graph_system.nodes._progress import emit_step

    agent_key = str(state.get("created_agent_key") or "").strip()
    if not agent_key:
        msg = "Verification failed — no agent_key recorded."
        return {
            "final_response": msg,
            "assistant_draft": msg,
        }

    emit_step(agent_key, 3, "running")
    issues: list[str] = []

    try:
        load_agent(agent_key)
    except Exception as exc:
        issues.append(f"YAML load failed: {exc}")

    for label, path_fn in (
        ("PERSONA.md", get_persona_path),
        ("SKILL.md", get_skill_path),
        ("TOOLS.md", get_tools_path),
    ):
        if not path_fn(agent_key).exists():
            issues.append(f"{label} missing")

    if issues:
        msg = f"Agent '{agent_key}' created with issues: {'; '.join(issues)}."
        emit_step(agent_key, 3, "failed")
    else:
        msg = (
            f"Agent '{agent_key}' created successfully. "
            "YAML config and all three definition files are in place."
        )
        emit_step(agent_key, 3, "completed")

    return {
        "final_response": msg,
        "assistant_draft": msg,
    }
