from typing import Any, Literal, TypedDict, Annotated
import operator


class ToolCallState(TypedDict):
    action: Literal["use_tool", "respond"]
    thought: str | None
    tool_stage: Literal["context", "artifact"] | None
    tool_name: str | None
    tool_input: dict[str, Any] | None
    final_answer: str | None
    approval_preview: dict[str, Any] | None


class SourceManifestEntry(TypedDict, total=False):
    id: str
    name: str
    uri: str
    mime_type: str
    status: Literal["listed", "reading", "read", "failed"]
    read_attempts: int


class SourceManifest(TypedDict, total=False):
    sources: list[SourceManifestEntry]


class AgentState(TypedDict, total=False):
    agent_key: str
    user_input: str
    config: dict[str, Any]
    metadata: dict[str, Any]
    persona_source_paths: list[str]
    messages: Annotated[list[dict[str, Any]], operator.add]
    llm_prompt_messages: list[dict[str, Any]]
    pending_tool_preview: dict[str, Any]
    interrupt_payload: dict[str, Any]
    interrupt_decision: str
    max_retries: int

    # --- Redesign-canonical durable keys (per AGENT_GRAPH_REDESIGN.md) ---
    phase: Literal["source", "artifact", "synthesis", "final"]
    source_manifest: SourceManifest
    # source_evidence is append-only. Nodes must return ONLY new entries
    # (a list of one-or-more entries to append), never the full prior list.
    source_evidence: Annotated[list[dict[str, Any]], operator.add]
    artifact_plan: dict[str, Any]
    artifact_input: dict[str, Any]
    artifact_result: dict[str, Any]
    synthesis_draft: str
    failure_type: Literal[
        "missing_source",
        "bad_artifact",
        "not_executed",
        "bad_synthesis",
        "retry_limit",
    ] | None
    validation_report: dict[str, Any]
    final_response: str

    # --- Routing dispatch keys ---
    router_next: str

    # --- Retry counters (per redesign: scoped per loop) ---
    source_attempt: int
    artifact_attempt: int
    synthesis_attempt: int

    # --- Legacy/transitional keys (to be removed as nodes migrate) ---
    # Decision/tool-call shared scratch
    active_tool_call: ToolCallState
    decision: dict[str, Any]
    tool_use: bool
    tool_call_history: list[dict[str, Any]]
    tool_turn_count: int
    attempt: int
    # Execution scratch
    execution_result: dict[str, Any]
    context_evidence: list[dict[str, Any]]
    latest_output: Any
    # Reflection/validation scratch
    reflection_result: dict[str, Any]
    source_reflection: dict[str, Any]
    source_execution_result: dict[str, Any]
    retry_reason: str | None
    draft_output: str
    finalization_reason: str
    router_origin: str
    # Artifact scratch
    artifact_generation_messages: list[dict[str, str]]
