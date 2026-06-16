from __future__ import annotations

from typing import Any

from solidcue.agent_configs import loader as _loader
from solidcue.core.graph_definition.state.schema import DefinitionState

_TARGET_TO_SAVE_FN = {
    "persona": "save_agent_persona",
    "skill":   "save_agent_skill",
    "tools":   "save_agent_tools",
}


def write_node(state: DefinitionState) -> dict[str, Any]:
    """Write definition_content to disk via the loader (overwrite-aware)."""
    target = str(state.get("definition_target") or "").strip()
    agent_key = str(state.get("agent_key") or "").strip()
    content = str(state.get("definition_content") or "").strip()
    overwrite = bool(state.get("overwrite", False))

    fn_name = _TARGET_TO_SAVE_FN.get(target)
    if not fn_name or not agent_key:
        return {"definition_path": ""}

    save_fn = getattr(_loader, fn_name)
    path = save_fn(agent_key, content or None, overwrite=overwrite)
    return {"definition_path": str(path)}
