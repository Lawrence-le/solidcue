from __future__ import annotations

import json
from typing import Any


def _build_reshape_system_prompt() -> str:
    return """
You are the user-facing manager of a multi-agent workspace. Earlier in this
conversation, agents gathered data and you presented it to the user. The user is now
asking a follow-up that only RESHAPES or RE-PRESENTS that already-gathered data —
for example: reformat it, change the layout, add or drop a column, change units,
sort, filter, or convert to another format.

You are given the structured results from those earlier runs. Compose the new
response using ONLY that existing data — do NOT call tools, do NOT invent values,
and do NOT re-derive numbers that are not present.

Rules:
- Answer the user's new request directly, applying it to the existing data.
- Use the structured data as the source of truth. The previously rendered text is
  only for reference to how it was shown before.
- If the user asks for a field that is genuinely NOT present in the data, say so
  plainly and offer to fetch fresh data — do not fabricate it.
- Keep values consistent with the original snapshot; do not silently update them.
- Match the user's language and the requested format. Return the response text only.
  No JSON wrapper, no preamble.
""".strip()


def _format_prior_results(agent_results: list[dict[str, Any]] | None) -> str:
    if not isinstance(agent_results, list) or not agent_results:
        return "None"
    blocks: list[str] = []
    for index, result in enumerate(agent_results, start=1):
        if not isinstance(result, dict):
            continue
        agent_key = str(result.get("agent_key") or "").strip() or "unknown"
        sub_task = str(result.get("sub_task") or "").strip()
        status = str(result.get("status") or "completed").strip()
        output = str(result.get("output") or "").strip() or "(no rendered output)"
        data = result.get("data")
        try:
            data_str = json.dumps(data, ensure_ascii=False, default=str) if data else "(none)"
        except Exception:
            data_str = str(data)
        header = f"[{index}] agent={agent_key} status={status}"
        if sub_task:
            header += f"\nsub_task: {sub_task}"
        blocks.append(
            f"{header}\n"
            f"structured_data:\n{data_str}\n"
            f"previously_shown:\n{output}"
        )
    return "\n\n".join(blocks) if blocks else "None"


def build_router_reshape_messages(
    *,
    user_input: str,
    agent_results: list[dict[str, Any]] | None,
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    runtime_context = (
        "=== NEW USER REQUEST (reshape of existing data) ===\n"
        f"{(user_input or '').strip()}\n\n"
        "=== EXISTING RESULTS FROM EARLIER RUNS ===\n"
        f"{_format_prior_results(agent_results)}\n\n"
        "Apply the new request to the existing data and compose the response now."
    )
    return [
        {"role": "system", "content": _build_reshape_system_prompt()},
        {"role": "user", "content": runtime_context},
    ]
