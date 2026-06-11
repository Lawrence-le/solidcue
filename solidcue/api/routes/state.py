"""State inspection endpoints."""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from solidcue.api.schemas import (
    ConversationMetadataResponse,
    RunStatusResponse,
    StateSnapshotResponse,
)
from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    delete_conversation_state,
    get_conversation_interrupt_payload,
    load_conversation_metadata,
    get_latest_thread_id_for_conversation,
    build_state_snapshot,
    delete_thread_state,
    get_latest_thread_id,
    get_thread_interrupt_payload,
    is_conversation_resumable,
    list_agent_state_keys,
    load_live_state_for_conversation,
    resolve_checkpoint_db_path,
)
from solidcue.services.run_engine import get_thread_run_status, is_thread_resumable

router = APIRouter(prefix="/state", tags=["state"])


class ThreadSummary(BaseModel):
    conversation_id: str
    thread_id: str
    agent_key: str | None
    step_count: int


class ConversationSummary(BaseModel):
    conversation_id: str
    thread_id: str | None


@router.get("/threads", response_model=list[ThreadSummary])
def list_threads() -> list[ThreadSummary]:
    """Return recent thread summaries from the checkpoint DB.

    Reads agent_key and step count directly from the metadata JSON column
    without loading full LangGraph state, so this is fast for large DBs.
    """
    db_path = resolve_checkpoint_db_path()
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'")
        if cur.fetchone() is None:
            conn.close()
            return []
        cur.execute(
            """
            WITH ranked AS (
                SELECT
                    COALESCE(NULLIF(json_extract(metadata, '$.conversation_id'), ''), thread_id) AS conversation_id,
                    thread_id,
                    json_extract(metadata, '$.agent_key') AS agent_key,
                    COALESCE(json_extract(metadata, '$.step'), 0) AS step_count,
                    rowid,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(NULLIF(json_extract(metadata, '$.conversation_id'), ''), thread_id)
                        ORDER BY rowid DESC
                    ) AS rn
                FROM checkpoints
                WHERE checkpoint_ns = ''
            ),
            aggregated AS (
                SELECT
                    conversation_id,
                    MAX(step_count) AS max_step,
                    MAX(rowid) AS latest_rowid
                FROM ranked
                GROUP BY conversation_id
            )
            SELECT
                ranked.conversation_id,
                ranked.thread_id,
                ranked.agent_key,
                aggregated.max_step
            FROM ranked
            JOIN aggregated
              ON aggregated.conversation_id = ranked.conversation_id
             AND aggregated.latest_rowid = ranked.rowid
            WHERE ranked.rn = 1
            ORDER BY ranked.rowid DESC
            """,
            (),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            ThreadSummary(
                conversation_id=row[0],
                thread_id=row[1],
                agent_key=row[2] or None,
                step_count=int(row[3]) if row[3] is not None else 0,
            )
            for row in rows
        ]
    except Exception:
        return []


@router.get("/keys", response_model=list[str])
def state_keys() -> list[str]:
    return list_agent_state_keys()


@router.get("/latest-thread")
def latest_thread() -> dict[str, str | None]:
    return {"thread_id": get_latest_thread_id()}


@router.get("/conversations/{conversation_id}/latest-thread", response_model=ConversationSummary)
def latest_thread_for_conversation(conversation_id: str) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=conversation_id,
        thread_id=get_latest_thread_id_for_conversation(conversation_id),
    )


@router.get(
    "/conversations/{conversation_id}/metadata",
    response_model=ConversationMetadataResponse,
)
def conversation_metadata(conversation_id: str) -> ConversationMetadataResponse:
    payload = load_conversation_metadata(conversation_id)
    return ConversationMetadataResponse(
        conversation_id=payload.get("conversation_id", conversation_id),
        agent_key=payload.get("agent_key")
        if isinstance(payload.get("agent_key"), str)
        else None,
        worked_seconds=int(payload.get("worked_seconds") or 0),
        last_thread_id=payload.get("last_thread_id")
        if isinstance(payload.get("last_thread_id"), str)
        else None,
        last_run_id=payload.get("last_run_id")
        if isinstance(payload.get("last_run_id"), str)
        else None,
        last_run_status=payload.get("last_run_status")
        if isinstance(payload.get("last_run_status"), str)
        else None,
        created_at=payload.get("created_at")
        if isinstance(payload.get("created_at"), str)
        else None,
        updated_at=payload.get("updated_at")
        if isinstance(payload.get("updated_at"), str)
        else None,
    )


@router.get("/example", response_model=StateSnapshotResponse)
def example_snapshot(
    key: list[str] | None = Query(default=None),
    include_all: bool = Query(default=False),
) -> StateSnapshotResponse:
    state = build_state_snapshot(keys=key, include_all=include_all)
    return StateSnapshotResponse(state=state)


@router.get("/live/{thread_id}", response_model=StateSnapshotResponse)
async def live_snapshot(
    thread_id: str,
    key: list[str] | None = Query(default=None),
    include_all: bool = Query(default=False),
) -> StateSnapshotResponse:
    state = await build_live_state_snapshot(
        thread_id=thread_id,
        keys=key,
        include_all=include_all,
    )
    return StateSnapshotResponse(thread_id=thread_id, state=state)


@router.get("/conversations/{conversation_id}/live", response_model=StateSnapshotResponse)
async def live_conversation_snapshot(
    conversation_id: str,
    key: list[str] | None = Query(default=None),
    include_all: bool = Query(default=False),
) -> StateSnapshotResponse:
    state = await load_live_state_for_conversation(conversation_id)
    if key and not include_all:
        schema_keys = set(state.keys())
        selected = [item for item in key if item in schema_keys]
        filtered_state = {item: state.get(item) for item in selected}
    else:
        filtered_state = state
    return StateSnapshotResponse(thread_id=get_latest_thread_id_for_conversation(conversation_id), state=filtered_state)


@router.get("/resumable/{thread_id}")
async def thread_resumable(thread_id: str) -> dict:
    """Return whether a thread has unfinished execution that can be continued."""
    return await is_thread_resumable(thread_id)


@router.get("/conversations/{conversation_id}/resumable")
async def conversation_resumable(conversation_id: str) -> dict:
    return await is_conversation_resumable(conversation_id)


@router.get("/interrupt/{thread_id}")
async def thread_interrupt(thread_id: str) -> dict:
    payload = await get_thread_interrupt_payload(thread_id)
    return {"interrupt": payload}


@router.get("/conversations/{conversation_id}/interrupt")
async def conversation_interrupt(conversation_id: str) -> dict:
    payload = await get_conversation_interrupt_payload(conversation_id)
    return {"interrupt": payload}


@router.get("/conversations/{conversation_id}/runs", response_model=RunStatusResponse)
def conversation_run_status(conversation_id: str) -> RunStatusResponse:
    thread_id = get_latest_thread_id_for_conversation(conversation_id)
    payload = get_thread_run_status(thread_id) if thread_id else {
        "thread_id": None,
        "run_id": None,
        "agent_key": None,
        "status": "idle",
        "error": None,
        "updated_at": None,
    }
    return RunStatusResponse(
        thread_id=payload.get("thread_id") if isinstance(payload.get("thread_id"), str) else conversation_id,
        run_id=payload.get("run_id") if isinstance(payload.get("run_id"), str) else None,
        agent_key=payload.get("agent_key") if isinstance(payload.get("agent_key"), str) else None,
        status=str(payload.get("status") or "idle"),
        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        updated_at=payload.get("updated_at") if isinstance(payload.get("updated_at"), str) else None,
    )


@router.get("/runs/{thread_id}", response_model=RunStatusResponse)
def run_status(thread_id: str) -> RunStatusResponse:
    payload = get_thread_run_status(thread_id)
    return RunStatusResponse(
        thread_id=thread_id,
        run_id=payload.get("run_id") if isinstance(payload.get("run_id"), str) else None,
        agent_key=payload.get("agent_key") if isinstance(payload.get("agent_key"), str) else None,
        status=str(payload.get("status") or "idle"),
        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        updated_at=payload.get("updated_at") if isinstance(payload.get("updated_at"), str) else None,
    )


@router.delete("/threads/{thread_id}", status_code=204)
def delete_thread(thread_id: str) -> Response:
    deleted = delete_thread_state(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")
    return Response(status_code=204)


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> Response:
    deleted = delete_conversation_state(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")
    return Response(status_code=204)
