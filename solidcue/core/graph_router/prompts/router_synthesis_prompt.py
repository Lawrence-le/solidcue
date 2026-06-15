from __future__ import annotations

from typing import Any


def _build_synthesis_system_prompt() -> str:
    return """
You are the user-facing manager of a multi-agent workspace. One or more execution
agents have each completed a sub-task and returned their results to you. Your job is
to compose the single, cohesive final response the user sees.

Rules:
- Answer the user's original request directly, as one unified reply.
- Integrate the agents' results into a coherent whole. Do not paste them verbatim,
  do not label sections by agent name, and do not mention the internal delegation
  ("Agent X said...", "the worker returned...").
- If the agents disagree or some sub-task failed, reconcile what you can and be honest
  about what is missing, without exposing internal machinery.
- Match the user's language and keep the tone helpful and natural.
- Return the response text only. No JSON, no preamble.
""".strip()


def _format_agent_results(agent_results: list[dict[str, Any]] | None) -> str:
    if not isinstance(agent_results, list) or not agent_results:
        return "None"
    blocks: list[str] = []
    for index, result in enumerate(agent_results, start=1):
        if not isinstance(result, dict):
            continue
        agent_key = str(result.get("agent_key") or "").strip() or "unknown"
        sub_task = str(result.get("sub_task") or "").strip()
        status = str(result.get("status") or "completed").strip()
        output = str(result.get("output") or "").strip() or "(no output)"
        header = f"[{index}] agent={agent_key} status={status}"
        if sub_task:
            header += f"\nsub_task: {sub_task}"
        blocks.append(f"{header}\nresult:\n{output}")
    return "\n\n".join(blocks) if blocks else "None"


def build_router_synthesis_messages(
    *,
    user_input: str,
    agent_results: list[dict[str, Any]] | None,
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    runtime_context = (
        "=== ORIGINAL USER REQUEST ===\n"
        f"{(user_input or '').strip()}\n\n"
        "=== AGENT RESULTS ===\n"
        f"{_format_agent_results(agent_results)}\n\n"
        "Compose the final user-facing response now."
    )
    return [
        {"role": "system", "content": _build_synthesis_system_prompt()},
        {"role": "user", "content": runtime_context},
    ]
