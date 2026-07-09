from __future__ import annotations

import json
import logging
from typing import Any

from solidcue.core.graph_definition.state.schema import DefinitionState
from solidcue.core.utils.metrics import timed_async_stream_generate
from solidcue.tools import playbook_registry

logger = logging.getLogger(__name__)


async def _tool_playbook_grounding(agent_spec: dict[str, Any]) -> str:
    """Collect the server playbooks for this agent's tools, as TOOLS.md grounding.

    Returns an instruction block carrying the authoritative tool-sequencing guidance,
    or "" when the agent has no tools / no server exposes a playbook / servers are
    unreachable. Best-effort: never raises into generation.
    """
    tools = agent_spec.get("selected_tools") or agent_spec.get("tools") or []
    tool_keys = [str(t).strip() for t in tools if str(t).strip()]
    if not tool_keys:
        return ""

    try:
        await playbook_registry.ensure_playbooks_warmed()
    except Exception:
        logger.warning("generate_node: playbook warm failed; proceeding without grounding")
        return ""

    seen: set[str] = set()
    blocks: list[str] = []
    for key in tool_keys:
        text = playbook_registry.get_playbook_for_tool(key)
        if text and text not in seen:
            seen.add(text)
            blocks.append(text)

    if not blocks:
        return ""

    joined = "\n\n".join(blocks)
    return (
        "=== TOOL PLAYBOOK (authoritative tool sequencing) ===\n"
        "The following playbook(s) describe how this agent's tools must be sequenced — "
        "data-dependencies (which call's result feeds the next), ordering, formats, and "
        "preconditions. Treat it as the source of truth for tool mechanics. In the TOOLS.md "
        "you write, follow these sequences exactly; keep the file focused on THIS agent's "
        "tools, paths, and workflow, and rely on the playbook for the mechanics rather than "
        "re-deriving them.\n\n"
        f"{joined}"
    )


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
    user_content = f"Create the {target} definition file for this agent:\n\n{spec_text}"

    # Ground the TOOLS.md target in the servers' tool playbooks so the writer follows
    # real tool sequences instead of improvising them. Other targets are untouched.
    if target == "tools":
        grounding = await _tool_playbook_grounding(agent_spec)
        if grounding:
            user_content = f"{user_content}\n\n{grounding}"

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
            "content": user_content,
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
