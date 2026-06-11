from __future__ import annotations

import json
from typing import Any

from solidcue.core.graph_agent.prompts.final_output_system_prompt import build_final_output_system_prompt


def build_final_output_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_final_output_system_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
    ]

