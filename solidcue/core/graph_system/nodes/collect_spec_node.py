from __future__ import annotations

import os
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from solidcue.core.graph_system.state.schema import SystemState
from solidcue.services.agent_service import CreateAgentInput

# Roles that must have a provider for a runnable agent (writer is optional).
_REQUIRED_PROVIDER_ROLES = ("decision", "lite", "reviewer")

# Provider roles and per-role fields the frontend form must render. `writer` is
# optional (CreateAgentInput defaults it to None); the rest are required.
_PROVIDER_ROLES = ("decision", "lite", "reviewer", "writer")
_PROVIDER_FIELDS = ("provider_type", "base_url", "model", "temperature", "api_key")
_SECRET_FIELDS = ("api_key",)
_PROVIDER_TYPES = ("anthropic", "openai", "openrouter")


def _validate_spec(agent_spec: dict[str, Any]) -> list[str]:
    """Return spec fields that are missing or invalid for CreateAgentInput.

    Empty list means the spec is complete and the agent can be created. Driving
    this off the pydantic model keeps the required set in one place and avoids a
    hand-maintained field list that could drift from CreateAgentInput.
    """
    try:
        CreateAgentInput(**agent_spec)
    except ValidationError as exc:
        fields: list[str] = []
        for err in exc.errors():
            loc = err.get("loc") or ()
            if not loc:
                continue
            name = str(loc[0])
            if name not in fields:
                fields.append(name)
        return fields
    return []


def _elicitation_gaps(agent_spec: dict[str, Any]) -> list[str]:
    """Fields that are optional on ``CreateAgentInput`` (so CLI/REST still work)
    but that the interactive gate must still gather from the human.

    Pydantic validation alone never surfaces these — they have defaults, so they
    pass as "present" and would silently land in the YAML/MD unasked. This is the
    substance the definition writer needs (does the agent produce artifacts, where
    do they go, what are its key tasks). Returned as extra fields to ask for,
    alongside the structural ``invalid_fields``.
    """
    gaps: list[str] = []
    if agent_spec.get("produces_artifacts") is None:
        gaps.append("produces_artifacts")
    elif agent_spec.get("produces_artifacts") and not str(
        agent_spec.get("artifact_destination") or ""
    ).strip():
        gaps.append("artifact_destination")
    if not agent_spec.get("key_tasks"):
        gaps.append("key_tasks")
    return gaps


def _available_tool_keys() -> list[str]:
    try:
        from solidcue.tools.loader import list_tools

        return [
            str(getattr(t, "tool_key", "")).strip()
            for t in list_tools()
            if str(getattr(t, "tool_key", "")).strip()
        ]
    except Exception:
        return []


def _workspace_provider_defaults() -> dict[str, Any] | None:
    """Build provider fields for all required roles from the workspace provider.

    Lets a conversational create-agent flow complete with no form: the new agent
    inherits the already-configured workspace provider and its API key. Returns
    None if no workspace provider / key is available, so the form fallback fires.
    """
    try:
        from solidcue.user.loader import load_user_profile

        rp = load_user_profile().router_provider
    except Exception:
        rp = None
    if rp is None:
        return None

    api_key = os.getenv(str(getattr(rp, "api_key_env", "") or "")) or ""
    if not api_key:
        return None

    temperature = getattr(rp, "temperature", None)
    fields: dict[str, Any] = {}
    for role in _REQUIRED_PROVIDER_ROLES:
        fields[f"{role}_provider_type"] = str(getattr(rp, "type", "") or "")
        fields[f"{role}_base_url"] = getattr(rp, "base_url", None)
        fields[f"{role}_model"] = str(getattr(rp, "model", "") or "")
        fields[f"{role}_temperature"] = temperature if temperature is not None else 0.2
        fields[f"{role}_api_key"] = api_key
    fields.setdefault("selected_tools", [])
    return fields


def _apply_provider_defaults(agent_spec: dict[str, Any]) -> dict[str, Any]:
    """Fill missing provider fields by inheriting the workspace provider. The
    explicit spec always wins, so a fully-specified spec is untouched."""
    if not _validate_spec(agent_spec):
        return agent_spec
    inherited = _workspace_provider_defaults()
    if not inherited:
        return agent_spec
    merged = dict(inherited)
    merged.update(agent_spec)
    return merged


def _form_schema() -> dict[str, Any]:
    """Describe the create-agent form so the frontend can render it from the
    interrupt payload alone — including which fields are secret (password inputs)."""
    return {
        "basic": ["name", "agent_key", "description", "selected_tools"],
        # Definition (.md) substance — the writer's inputs. `definition` fields are
        # gathered every interactive create; `advanced` fields are optional YAML
        # overrides the frontend may render behind a disclosure.
        "definition": ["produces_artifacts", "artifact_destination", "key_tasks", "examples"],
        "advanced": ["allowed_tasks", "style", "constraints", "validation_policy"],
        "provider_roles": list(_PROVIDER_ROLES),
        "provider_fields": list(_PROVIDER_FIELDS),
        "secret_fields": list(_SECRET_FIELDS),
        "provider_types": list(_PROVIDER_TYPES),
        "available_tools": _available_tool_keys(),
    }


def collect_spec_node(state: SystemState) -> dict[str, Any]:
    """Validate ``agent_spec`` before agent creation.

    Option A (interrupt, frontend form): when the spec is incomplete or invalid
    for ``CreateAgentInput``, pause and emit a ``form_schema`` so the frontend can
    render a secure create-agent form (provider/key inputs included). On resume
    (``Command(resume={"agent_spec": {...}})``) the reply is merged and the spec
    is re-validated. The resume payload may be the spec fields directly or wrapped
    as ``{"agent_spec": {...}}``.

    Option B (pre-supplied): a complete spec (e.g. from ``POST /agents``) never
    interrupts and proceeds straight to creation.

    The gate re-asks until the spec is structurally valid — a single incomplete
    reply no longer errors out, it just prompts again. Elicitation gaps (the
    definition/MD substance) are surfaced on the first ask but never block, since
    the human may knowingly leave them empty.
    """
    agent_spec = _apply_provider_defaults(dict(state.get("agent_spec") or {}))

    invalid = _validate_spec(agent_spec)
    gaps = _elicitation_gaps(agent_spec)
    asked = False
    # Loop while structurally invalid, or on the very first pass if there are only
    # elicitation gaps to gather. After the first ask, gaps stop driving the loop
    # (they don't block), so only missing required fields keep it open.
    while invalid or (gaps and not asked):
        reply = interrupt(
            {
                "type": "collect_agent_spec",
                "agent_spec": agent_spec,
                "invalid_fields": invalid,
                "gather_fields": gaps,
                "form_schema": _form_schema(),
                "message": "Fill in the agent details and provider settings.",
            }
        )
        provided: Any = reply
        if isinstance(reply, dict) and isinstance(reply.get("agent_spec"), dict):
            provided = reply["agent_spec"]
        if isinstance(provided, dict):
            agent_spec.update(provided)
        asked = True
        invalid = _validate_spec(agent_spec)
        gaps = _elicitation_gaps(agent_spec)

    return {
        "agent_spec": agent_spec,
        "created_agent_key": str(agent_spec.get("agent_key", "")).strip(),
    }
