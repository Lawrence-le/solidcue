from typing import Any, Literal, TypedDict, Annotated
import operator


class ToolCallState(TypedDict):
    action: Literal["use_tool", "respond"]
    thought: str | None
    tool_name: str | None
    tool_input: dict[str, Any] | None
    approval_preview: dict[str, Any] | None


class AgentState(TypedDict, total=False):
    # --- Identity / request context ---
    agent_key: str
    thread_id: str
    conversation_id: str
    user_input: str
    config: dict[str, Any]
    metadata: dict[str, Any]

    # --- Message state ---
    messages: Annotated[list[dict[str, Any]], operator.add]
    chat_history: Annotated[list[dict[str, Any]], operator.add]
    llm_prompt_messages: list[dict[str, Any]]

    # --- Metrics ---
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

    # --- Redesign-canonical durable keys (per AGENT_GRAPH_REDESIGN.md) ---
    phase: Literal["source", "artifact", "synthesis", "final"]
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

    # --- Retry policy / counters (per redesign: scoped per loop) ---
    max_retries: int
    source_attempt: int
    artifact_attempt: int
    synthesis_attempt: int
    retry_reason: str | None

    # --- Decision/tool-call state (execution prep) ---
    active_tool_call: ToolCallState
    decision: dict[str, Any]
    tool_use: bool
    tool_call_history: list[dict[str, Any]]
    tool_turn_count: int

    # --- Execution state (source loop) ---
    execution_result: dict[str, Any]
    handoff: dict[str, Any]
