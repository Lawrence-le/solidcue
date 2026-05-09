from __future__ import annotations

import json
from typing import Any


def build_validation_system_prompt() -> str:
    return (
        "You are a strict response validator for an AI agent.\n"
        "Evaluate whether the draft output is acceptable as a final user-facing answer.\n"
        "Primary criteria:\n"
        "1) The answer addresses the user's request directly.\n"
        "2) The answer is grounded in available tool evidence and does not contradict it.\n"
        "3) The answer is not internal JSON, chain-of-thought, or tool orchestration metadata.\n"
        "4) If information is partial, the answer should clearly state what is known vs unknown.\n"
        "5) For artifact-producing requests (documents/files/sheets/pdfs/csv/etc.), reject summary-only answers "
        "when the requested artifact has not been produced yet.\n"
        "Return ONLY one JSON object with keys:\n"
        "- passed: boolean\n"
        "- reason: string\n"
        "- score: number between 0 and 1\n"
        "- retry_tag: \"none\" or \"artifact_required\"\n"
        "If retry_tag is \"artifact_required\", reason must begin with \"ARTIFACT_REQUIRED:\".\n"
        "Do not include markdown fences or extra text."
    )


def build_validation_messages(
    *,
    user_query: str,
    draft_output: str,
    decision: dict[str, Any] | None = None,
    execution_result: dict[str, Any] | None = None,
    tool_call_history: list[dict[str, Any]] | None = None,
    retry_reason: str | None = None,
    tool_turn_count: int | None = None,
) -> list[dict[str, str]]:
    payload = {
        "user_query": user_query,
        "draft_output": draft_output,
        "decision": decision if isinstance(decision, dict) else {},
        "execution_result": execution_result if isinstance(execution_result, dict) else {},
        "tool_call_history": tool_call_history if isinstance(tool_call_history, list) else [],
        "retry_reason": retry_reason or "",
        "tool_turn_count": tool_turn_count if isinstance(tool_turn_count, int) else 0,
    }

    return [
        {"role": "system", "content": build_validation_system_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
    ]
