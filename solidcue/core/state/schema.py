from typing import Any, Literal, TypedDict, Annotated
import operator


class ToolCallState(TypedDict):
    action: Literal["use_tool", "respond"]
    thought: str | None
    tool_name: str | None
    tool_input: dict[str, Any] | None
    approval_preview: dict[str, Any] | None


class AgentState(TypedDict, total=False):
    agent_key: str
    user_input: str
    config: dict[str, Any]
    metadata: dict[str, Any]
    messages: Annotated[list[dict[str, Any]], operator.add]
    llm_prompt_messages: list[dict[str, Any]]
    metric_usage_events: Annotated[list[dict[str, Any]], operator.add]
    metric_classifier: dict[str, Any]
    metric_planning: dict[str, Any]
    metric_decision: dict[str, Any]
    metric_discovery: dict[str, Any]
    metric_synthesis: dict[str, Any]
    metric_reflection: dict[str, Any]
    metric_validation: dict[str, Any]
    metric_validation_hhem: dict[str, Any]
    metric_final_output: dict[str, Any]
    max_retries: int

    # --- Redesign-canonical durable keys (per AGENT_GRAPH_REDESIGN.md) ---
    phase: Literal["source", "artifact", "synthesis", "final", "conversational"]
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

    # --- Task planning ---
    task_plan: list[dict[str, Any]]
    current_task: str

    # --- Routing dispatch keys ---
    router_next: str

    # --- Retry counters (per redesign: scoped per loop) ---
    source_attempt: int
    artifact_attempt: int
    synthesis_attempt: int

    # --- Decision/tool-call state (execution prep) ---
    active_tool_call: ToolCallState
    decision: dict[str, Any]
    tool_use: bool
    tool_call_history: list[dict[str, Any]]
    tool_turn_count: int

    # --- Execution state (source loop) ---
    execution_result: dict[str, Any]
    context_evidence: list[dict[str, Any]]
    handoff: dict[str, Any]

    # --- Prompt generation (for context to LLM nodes) ---
    retry_reason: str | None
