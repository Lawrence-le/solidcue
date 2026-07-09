"""select_tools_node — choose which registry tools a new agent needs.

graph_system's first LLM node. Selection is coarse-to-fine so the model reasons
about *capabilities and how tools compose*, not a flat list of one-liners:

  1. list servers (each with a purpose + its tools)
  2. LLM picks the relevant server(s) for the agent's goal        (call #1)
  3. read those servers' playbooks (how their tools chain)
  4. LLM picks the exact tools, seeing only the picked servers'
     tools + their playbooks + always-available non-MCP tools     (call #2)

The result is written into agent_spec.selected_tools. Falls back to a flat
single-pass pick if server selection yields nothing, so it never regresses to
empty. Traced via timed_async_stream_generate, same as graph_definition.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from solidcue.core.graph_system.state.schema import SystemState
from solidcue.core.utils.metrics import timed_async_stream_generate
from solidcue.tools import playbook_registry
from solidcue.tools.loader import list_tools, load_mcp_server

logger = logging.getLogger(__name__)


def _get_workspace_provider() -> Any:
    try:
        from solidcue.core.graph_router.nodes._shared import _PROFILE_ROUTER_PROVIDER

        return _PROFILE_ROUTER_PROVIDER
    except Exception:
        return None


def _server_catalog() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return (servers, non_mcp_tools).

    servers: [{server_key, name, purpose, tools: [{tool_key, description}]}]
    non_mcp_tools: [{tool_key, description}] — utility tools with no server/playbook.
    """
    servers: dict[str, dict[str, Any]] = {}
    non_mcp: list[dict[str, str]] = []
    try:
        for tool in list_tools():
            tool_key = str(getattr(tool, "tool_key", "") or "").strip()
            if not tool_key or not getattr(tool, "enabled", True):
                continue
            description = str(getattr(tool, "description", "") or "").strip()
            mcp = getattr(tool, "mcp", None)
            server_key = getattr(mcp, "server_key", None) if tool.type == "mcp" else None
            if server_key:
                bucket = servers.get(server_key)
                if bucket is None:
                    try:
                        cfg = load_mcp_server(server_key)
                        name = cfg.name
                        purpose = (getattr(cfg, "purpose", "") or cfg.description or "").strip()
                    except Exception:
                        name, purpose = server_key, ""
                    bucket = {"server_key": server_key, "name": name, "purpose": purpose, "tools": []}
                    servers[server_key] = bucket
                bucket["tools"].append({"tool_key": tool_key, "description": description})
            else:
                non_mcp.append({"tool_key": tool_key, "description": description})
    except Exception:
        logger.exception("select_tools: failed to build server catalog")
        return [], []
    return list(servers.values()), non_mcp


def _goal_text(agent_spec: dict[str, Any]) -> str:
    """Full goal for tool selection: description + key tasks + artifact destination.

    The bare `description` often omits the output/save requirement (e.g. "save to
    Google Drive"), which is exactly what determines whether a save/upload chain is
    needed. Fold in `key_tasks` and `artifact_destination` so selection sees the
    whole job, not just the summary line.
    """
    parts: list[str] = []
    description = str(agent_spec.get("description") or "").strip()
    if description:
        parts.append(description)

    tasks = agent_spec.get("key_tasks")
    if isinstance(tasks, (list, tuple)):
        joined = "; ".join(str(t).strip() for t in tasks if str(t).strip())
    else:
        joined = str(tasks or "").strip()
    if joined:
        parts.append(f"Key tasks: {joined}")

    if agent_spec.get("produces_artifacts"):
        dest = str(agent_spec.get("artifact_destination") or "").strip()
        parts.append(
            f"Produces a saved output artifact. Destination: {dest}"
            if dest
            else "Produces a saved output artifact."
        )

    return "\n".join(parts)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _pick_servers(
    provider: Any, name: str, description: str, servers: list[dict[str, Any]]
) -> list[str]:
    """Stage A: pick the relevant server_keys for the agent's goal."""
    valid = {s["server_key"] for s in servers}
    lines = []
    for s in servers:
        tool_names = ", ".join(t["tool_key"] for t in s["tools"])
        purpose = s["purpose"] or "(no description)"
        lines.append(f"- {s['server_key']}: {purpose}\n    tools: {tool_names}")
    catalog = "\n".join(lines)

    messages = [
        {
            "role": "system",
            "content": (
                "You choose which tool servers a new agent needs. Each server groups related "
                "tools. Pick EVERY server whose tools the agent needs to fulfil its purpose "
                "end-to-end — including servers needed only for an intermediate step (for "
                "example, a server that generates a file before another server uploads it). "
                "Prefer completeness over minimalism. Return only server_keys."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent name: {name}\n"
                f"Agent goal:\n{description}\n\n"
                f"Available servers:\n{catalog}\n\n"
                'Return JSON only: {"servers": ["server_key", ...]}.'
            ),
        },
    ]
    try:
        output, _metric = await timed_async_stream_generate(provider, messages, node_name="select_servers")
        parsed = _extract_json_object(output)
        raw = parsed.get("servers") if isinstance(parsed, dict) else None
        if isinstance(raw, list):
            return [str(s).strip() for s in raw if str(s).strip() in valid]
    except Exception:
        logger.exception("select_tools: server selection failed")
    return []


async def _pick_tools(
    provider: Any,
    name: str,
    description: str,
    candidate_tools: list[dict[str, str]],
    playbooks: list[str],
) -> list[str]:
    """Stage B: pick the exact tool_keys from the scoped candidate set."""
    valid = {t["tool_key"] for t in candidate_tools}
    tool_lines = "\n".join(f"- {t['tool_key']}: {t['description']}" for t in candidate_tools)
    playbook_block = ""
    if playbooks:
        playbook_block = (
            "\n\nTool playbooks (how these tools chain — data-dependencies, ordering, "
            "preconditions). Use them to select every tool a required chain needs:\n\n"
            + "\n\n".join(playbooks)
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You select the exact tools a new agent needs, using exact tool_keys from the "
                "provided list only. Select the COMPLETE set required to fulfil the goal "
                "end-to-end, including every tool in a required chain (for example: generate a "
                "file, resolve its destination folder, then upload it). Follow the playbooks for "
                "how tools chain. Do not add unrelated tools; do not invent tool_keys."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent name: {name}\n"
                f"Agent goal:\n{description}\n\n"
                f"Available tools:\n{tool_lines}"
                f"{playbook_block}\n\n"
                'Return JSON only: {"selected_tools": ["tool_key", ...]}. Use [] if none apply.'
            ),
        },
    ]
    try:
        output, _metric = await timed_async_stream_generate(provider, messages, node_name="select_tools")
        parsed = _extract_json_object(output)
        raw = parsed.get("selected_tools") if isinstance(parsed, dict) else None
        if isinstance(raw, list):
            return [str(t).strip() for t in raw if str(t).strip() in valid]
    except Exception:
        logger.exception("select_tools: tool selection failed")
    return []


async def select_tools_node(state: SystemState) -> dict[str, Any]:
    from solidcue.core.graph_system.nodes._progress import emit_plan, emit_step

    agent_spec = dict(state.get("agent_spec") or {})
    agent_key = str(state.get("created_agent_key") or agent_spec.get("agent_key") or "")
    emit_plan(agent_key)
    emit_step(agent_key, 0, "running")

    servers, non_mcp = _server_catalog()
    all_keys = {t["tool_key"] for s in servers for t in s["tools"]} | {t["tool_key"] for t in non_mcp}

    # Respect tools already chosen upstream (e.g. a pre-supplied spec).
    existing = [
        str(t).strip()
        for t in (agent_spec.get("selected_tools") or [])
        if str(t).strip() in all_keys
    ]
    if existing:
        agent_spec["selected_tools"] = existing
        emit_step(agent_key, 0, "completed")
        return {"agent_spec": agent_spec}

    provider = _get_workspace_provider()
    if not all_keys or provider is None:
        agent_spec["selected_tools"] = []
        emit_step(agent_key, 0, "completed")
        return {"agent_spec": agent_spec}

    name = str(agent_spec.get("name") or "").strip()
    description = _goal_text(agent_spec) or str(agent_spec.get("description") or "").strip()

    # Stage A: pick servers.
    picked_servers = await _pick_servers(provider, name, description, servers) if servers else []

    # Stage B candidates: picked servers' tools + always-available non-MCP tools.
    # Fall back to ALL tools if server selection produced nothing, so we never regress.
    if picked_servers:
        picked_set = set(picked_servers)
        candidate_tools = [t for s in servers if s["server_key"] in picked_set for t in s["tools"]]
        candidate_tools += non_mcp
        try:
            await playbook_registry.ensure_playbooks_warmed()
        except Exception:
            logger.warning("select_tools: playbook warm failed; selecting without playbooks")
        seen: set[str] = set()
        playbooks: list[str] = []
        for sk in picked_servers:
            text = playbook_registry.get_server_playbook(sk)
            if text and text not in seen:
                seen.add(text)
                playbooks.append(text)
    else:
        candidate_tools = [t for s in servers for t in s["tools"]] + non_mcp
        playbooks = []

    selected = await _pick_tools(provider, name, description, candidate_tools, playbooks)

    agent_spec["selected_tools"] = selected
    emit_step(agent_key, 0, "completed")
    return {"agent_spec": agent_spec}
