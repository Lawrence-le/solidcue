from __future__ import annotations

from typing import Any

from solidcue.agent_configs.loader import SKILLS_ROOT_DIR
from solidcue.core.graph_definition.state.schema import DefinitionState


def load_contract_node(state: DefinitionState) -> dict[str, Any]:
    """Load the create-<target>.md skill contract into contract_skill."""
    target = str(state.get("definition_target") or "").strip()
    if not target:
        return {"contract_skill": ""}

    contract_path = SKILLS_ROOT_DIR / f"create-{target}.md"
    if not contract_path.exists():
        return {"contract_skill": ""}

    return {"contract_skill": contract_path.read_text(encoding="utf-8").strip()}
