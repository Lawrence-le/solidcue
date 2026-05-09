from datetime import datetime, timezone
from typing import Any

from solidcue.core.state.schema import AgentState


def initialize_node(state: AgentState) -> dict[str, Any]:
    """Initialize missing state fields with safe defaults."""
    metadata = dict(state.get("metadata", {}))
    if "current_time_utc" not in metadata:
        metadata["current_time_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "metadata": metadata,
        "persona_source_paths": list(state.get("persona_source_paths", [])),
        "messages": list(state.get("messages", [])),
        "llm_prompt_messages": list(state.get("llm_prompt_messages", [])),
        "max_retries": int(state.get("max_retries", 0)),
        "phase": state.get("phase") or "source",
        "source_manifest": state.get("source_manifest") or {"sources": []},
        # source_evidence uses operator.add reducer; return [] as no-op to avoid duplication on replay.
        "source_evidence": [],
        "failure_type": state.get("failure_type"),
        "source_attempt": int(state.get("source_attempt", 0)),
        "artifact_attempt": int(state.get("artifact_attempt", 0)),
        "synthesis_attempt": int(state.get("synthesis_attempt", 0)),
    }
