from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any

from solidcue.core.graph_router.nodes._shared import normalize_text
from solidcue.core.graph_router.state.schema import RouterState
from solidcue.user.loader import load_user_profile


def _resolve_router_metadata(state: RouterState) -> dict[str, Any]:
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
        metadata["current_time"] = now_local.strftime("%H:%M:%S")
    if "current_date" not in metadata:
        metadata["current_date"] = now_local.strftime("%A, %B %d, %Y")
    if "location" not in metadata:
        location = config_dict.get("location")
        if not isinstance(location, str) or not location.strip():
            profile_location = load_user_profile().location
            location = profile_location if isinstance(profile_location, str) and profile_location.strip() else ""
        metadata["location"] = location if isinstance(location, str) and location.strip() else "Unknown location"

    return metadata


def initialize_router_node(state: RouterState) -> dict[str, Any]:
    conversation_id = normalize_text(state.get("conversation_id")) or normalize_text(
        state.get("thread_id")
    )
    updates: dict[str, Any] = {}
    if conversation_id:
        updates["conversation_id"] = conversation_id
    updates["worked_seconds"] = int(state.get("worked_seconds") or 0)
    updates["timer_started_at"] = state.get("timer_started_at")
    updates["metadata"] = _resolve_router_metadata(state)

    # Append the user's turn to the persisted chat_history channel (operator.add).
    # The LangGraph Server thread checkpoint accumulates history across turns, so
    # reading state["chat_history"] in downstream nodes gives the full conversation
    # history for this thread.
    user_input = normalize_text(state.get("user_input"))
    if user_input:
        updates["chat_history"] = [{"role": "user", "content": user_input}]

    return updates
