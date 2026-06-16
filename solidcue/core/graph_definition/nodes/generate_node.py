from __future__ import annotations

import json
import logging
from typing import Any

from solidcue.core.graph_definition.state.schema import DefinitionState
from solidcue.core.utils.metrics import timed_async_stream_generate

logger = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    """Remove a wrapping ``` fence the model often adds around the whole file
    (e.g. ```markdown ... ```), so the raw markdown is written, not a fenced block."""
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    # Drop the opening fence line (``` or ```markdown / ```md).
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    # Drop the closing fence line if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _get_workspace_provider() -> Any:
    try:
        from solidcue.core.graph_router.nodes._shared import _PROFILE_ROUTER_PROVIDER
        return _PROFILE_ROUTER_PROVIDER
    except Exception:
        return None


async def generate_node(state: DefinitionState) -> dict[str, Any]:
    """Call the workspace provider to generate definition content from the contract + spec."""
    target = str(state.get("definition_target") or "").strip()
    contract_skill = str(state.get("contract_skill") or "").strip()
    agent_spec = state.get("agent_spec") or {}

    spec_text = json.dumps(agent_spec, indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical writer creating agent definition files for a "
                "generic AI agent framework. Follow the contract exactly.\n\n"
                f"{contract_skill}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create the {target} definition file for this agent:\n\n{spec_text}"
            ),
        },
    ]

    provider = _get_workspace_provider()
    if provider is None:
        logger.warning("No workspace provider configured; skipping LLM generation for %s", target)
        return {"definition_content": ""}

    # Wrap the provider call in a Langfuse generation span — same mechanism the
    # agent graph uses (metrics.timed_async_stream_generate), so the persona/skill/
    # tools generation shows up as a traced generation.
    output = ""
    try:
        output, _metric = await timed_async_stream_generate(
            provider, messages, node_name=f"generate_{target}"
        )
    except Exception:
        logger.exception("generate_node failed for target=%s", target)

    return {"definition_content": _strip_code_fence(output)}
