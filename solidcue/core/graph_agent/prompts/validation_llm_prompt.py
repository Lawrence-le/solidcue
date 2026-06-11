from __future__ import annotations

import json
from typing import Any

from solidcue.core.graph_agent.prompts.validation_llm_system_prompt import build_validation_llm_system_prompt


def build_validation_messages(
    *,
    user_query: str,
    draft_output: str,
    validation_evidence: list[dict[str, Any]] | None = None,
    task_description: str | None = None,
) -> list[dict[str, str]]:
    payload: dict[str, Any] = {
        "user_query": user_query,
        "draft_output": draft_output,
        "validation_evidence": validation_evidence if isinstance(validation_evidence, list) else [],
    }
    if task_description:
        payload["current_task"] = task_description

    return [
        {"role": "system", "content": build_validation_llm_system_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
    ]
