"""State inspection endpoints."""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from solidcue.api.schemas import RunStatusResponse, StateSnapshotResponse
from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    build_state_snapshot,
    delete_thread_state,
    get_latest_thread_id,
    get_thread_interrupt_payload,
    list_agent_state_keys,
    resolve_checkpoint_db_path,
)
from solidcue.services.run_engine import get_thread_run_status, is_thread_resumable

router = APIRouter(prefix="/state", tags=["state"])


class ThreadSummary(BaseModel):
    thread_id: str
    agent_key: str | None
    step_count: int


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
        cur.execute(
            """
            SELECT
                thread_id,
                json_extract(metadata, '$.agent_key') AS agent_key,
                MAX(json_extract(metadata, '$.step'))  AS max_step
            FROM checkpoints
            WHERE checkpoint_ns = ''
            GROUP BY thread_id
            ORDER BY MAX(rowid) DESC
            """,
            (),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            ThreadSummary(
                thread_id=row[0],
                agent_key=row[1] or None,
                step_count=int(row[2]) if row[2] is not None else 0,
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


@router.get("/resumable/{thread_id}")
async def thread_resumable(thread_id: str) -> dict:
    """Return whether a thread has unfinished execution that can be continued."""
    return await is_thread_resumable(thread_id)


@router.get("/interrupt/{thread_id}")
async def thread_interrupt(thread_id: str) -> dict:
    payload = await get_thread_interrupt_payload(thread_id)
    return {"interrupt": payload}


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
