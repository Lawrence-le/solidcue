import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any

from dotenv import load_dotenv

from solidcue.services.hhem_service import load_hhem_model
from solidcue.core.state.schema import AgentState
from solidcue.observability import get_env_path

"""
Initialize Node - Function Overview
-----------------------------------

_is_truthy:
Interpret env toggles.

initialize_node:
Bootstrap missing state defaults (metadata, retry counters, phase, timestamps).
"""


load_dotenv(dotenv_path=get_env_path())


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def initialize_node(state: AgentState) -> dict[str, Any]:
    """Initialize missing state fields with safe defaults."""
    # Phase 1: optional warmup/preload.
    if _is_truthy(os.getenv("SOLIDCUE_HHEM_PRELOAD")):
        load_hhem_model()
    # Phase 2: resolve timezone + metadata defaults.
    metadata = dict(state.get("metadata", {}))
    config = state.get("config")
    config_dict = config if isinstance(config, dict) else {}

    tz_name = metadata.get("timezone")
    if not isinstance(tz_name, str) or not tz_name.strip():
        tz_name = config_dict.get("timezone")
    if not isinstance(tz_name, str) or not tz_name.strip():
        tz_name = os.getenv("SOLIDCUE_DEFAULT_TIMEZONE")
    if not isinstance(tz_name, str) or not tz_name.strip():
        tz_name = "UTC"

    tz_for_now = timezone.utc
    try:
        tz_for_now = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        metadata["timezone"] = "UTC"
    else:
        metadata["timezone"] = tz_name

    now_local = datetime.now(tz_for_now)
    if "current_time" not in metadata:
        metadata["current_time"] = now_local.strftime("%A, %B %d, %Y %H:%M:%S")
    if "current_date" not in metadata:
        metadata["current_date"] = now_local.strftime("%Y-%m-%d")
    if "location" not in metadata:
        location = config_dict.get("location")
        metadata["location"] = location if isinstance(location, str) and location.strip() else "Unknown location"
    if "current_time_utc" not in metadata:
        metadata["current_time_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Phase 3: normalize retry limits and core runtime fields.
    raw_max_retries = state.get("max_retries")
    max_retries = raw_max_retries if isinstance(raw_max_retries, int) and raw_max_retries >= 0 else 3

    return {
        "metadata": metadata,
        "messages": list(state.get("messages", [])),
        "llm_prompt_messages": list(state.get("llm_prompt_messages", [])),
        "max_retries": max_retries,
        "phase": state.get("phase") or "source",
        "failure_type": state.get("failure_type"),
        "source_attempt": int(state.get("source_attempt", 0)),
        "artifact_attempt": int(state.get("artifact_attempt", 0)),
        "synthesis_attempt": int(state.get("synthesis_attempt", 0)),
    }
