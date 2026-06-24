from __future__ import annotations

import json
from typing import Any


def _build_synthesis_system_prompt() -> str:
    return """
You are the user-facing manager of a multi-agent workspace. One or more execution
agents have each completed a sub-task and returned their results to you. Your job is
to compose the single, cohesive final response the user sees.

The AGENT RESULTS below include BOTH results from this turn and results retained from
earlier in the session. Each result carries a structured `data` block (the actual
values gathered) and a rendered `result` text.

Rules:
- Answer the user's original request directly, as one unified reply.
- Use `data` as the source of truth for values. When the request extends or updates an
  earlier answer (e.g. "also add X", "include Y"), COMBINE the new result with the
  relevant retained results into one complete answer — do not ask the user for data
  that is already present in any result's `data` block.
- A result's rendered `result` text may say it lacks data from other sub-tasks; ignore
  that — you have all the results here, so reconcile them yourself.
- Integrate the results into a coherent whole. Do not paste them verbatim, do not label
  sections by agent name, and do not mention the internal delegation.
- Only state something is missing if it is absent from EVERY result's `data` block.
- CHAT_HISTORY is provided for conversational continuity only — use it to match the
  format, tone, and references the user established earlier (e.g. "the same table").
  It is NOT a data source: take all values from the `data` blocks, never from history.
- Match the user's language and keep the tone helpful and natural.
- Return the response text only. No JSON, no preamble.
""".strip()


def _format_chat_history(chat_history: list[dict[str, str]] | None, *, limit: int = 6) -> str:
    if not isinstance(chat_history, list) or not chat_history:
        return "None"
    lines: list[str] = []
    for entry in chat_history[-limit:]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip()
        content = str(entry.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None"


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
        data = result.get("data")
        try:
            data_str = json.dumps(data, ensure_ascii=False, default=str) if data else "(none)"
        except Exception:
            data_str = str(data)
        header = f"[{index}] agent={agent_key} status={status}"
        if sub_task:
            header += f"\nsub_task: {sub_task}"
        blocks.append(f"{header}\ndata:\n{data_str}\nresult:\n{output}")
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
        "=== CHAT_HISTORY (for continuity/format only — not a data source) ===\n"
        f"{_format_chat_history(chat_history)}\n\n"
        "=== AGENT RESULTS (source of truth for values) ===\n"
        f"{_format_agent_results(agent_results)}\n\n"
        "Compose the final user-facing response now."
    )
    return [
        {"role": "system", "content": _build_synthesis_system_prompt()},
        {"role": "user", "content": runtime_context},
    ]
