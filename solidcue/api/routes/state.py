"""State inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from solidcue.api.schemas import (
    ConversationMetadataResponse,
    RunStatusResponse,
    StateSnapshotResponse,
)
from solidcue.services.lg_client import (
    get_lg_client,
    get_lg_thread_by_conversation,
    get_lg_thread_status,
)
from solidcue.services.state_snapshot_service import (
    build_live_state_snapshot,
    delete_conversation_state,
    get_conversation_interrupt_payload,
    load_conversation_metadata,
    load_conversation_snapshot,
    get_latest_thread_id_for_conversation,
    build_state_snapshot,
    delete_thread_state,
    get_latest_thread_id,
    get_thread_interrupt_payload,
    is_conversation_resumable,
    list_agent_state_keys,
    load_live_state_for_conversation,
)
from pydantic import BaseModel

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
async def list_threads() -> list[ThreadSummary]:
    """Return recent thread summaries from the LangGraph Server."""
    try:
        client = get_lg_client()
        # Search all threads; filter out internal worker threads.
        results = await client.threads.search(limit=100)
        summaries = []
        for thread in results:
            thread_id = thread.get("thread_id") if isinstance(thread, dict) else getattr(thread, "thread_id", None)
            metadata = thread.get("metadata") if isinstance(thread, dict) else getattr(thread, "metadata", None)
            if not thread_id:
                continue
            conversation_id = (metadata or {}).get("conversation_id") or thread_id
            # Skip internal orchestration worker threads.
            if "::worker::" in conversation_id:
                continue
            summaries.append(
                ThreadSummary(
                    conversation_id=conversation_id,
                    thread_id=thread_id,
                    agent_key=None,
                    step_count=0,
                )
            )
        return summaries
    except Exception:
        return []


@router.get("/keys", response_model=list[str])
def state_keys() -> list[str]:
    return list_agent_state_keys()


@router.get("/latest-thread")
async def latest_thread() -> dict[str, str | None]:
    return {"thread_id": await get_latest_thread_id()}


@router.get("/conversations/{conversation_id}/latest-thread", response_model=ConversationSummary)
async def latest_thread_for_conversation(conversation_id: str) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=conversation_id,
        thread_id=await get_latest_thread_id_for_conversation(conversation_id),
    )


@router.get(
    "/conversations/{conversation_id}/metadata",
    response_model=ConversationMetadataResponse,
)
async def conversation_metadata(conversation_id: str) -> ConversationMetadataResponse:
    payload = load_conversation_metadata(conversation_id)
    live_state = await load_live_state_for_conversation(conversation_id)
    worked_seconds = live_state.get("worked_seconds")
    return ConversationMetadataResponse(
        conversation_id=payload.get("conversation_id", conversation_id),
        agent_key=payload.get("agent_key")
        if isinstance(payload.get("agent_key"), str)
        else None,
        worked_seconds=(
            int(worked_seconds)
            if isinstance(worked_seconds, (int, float))
            else 0
        ),
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
    return StateSnapshotResponse(thread_id=await get_latest_thread_id_for_conversation(conversation_id), state=filtered_state)


@router.get("/conversations/{conversation_id}/snapshot", response_model=StateSnapshotResponse)
async def conversation_snapshot(conversation_id: str) -> StateSnapshotResponse:
    state = await load_conversation_snapshot(conversation_id)
    thread_id = state.get("last_thread_id") if isinstance(state.get("last_thread_id"), str) else None
    return StateSnapshotResponse(thread_id=thread_id, state=state)


@router.get("/resumable/{thread_id}")
async def thread_resumable(thread_id: str) -> dict:
    """Return whether a LangGraph Server thread is in an interrupted/resumable state."""
    thread_status = await get_lg_thread_status(thread_id)
    resumable = thread_status in ("interrupted", "busy")
    return {"resumable": resumable, "next_nodes": [], "thread_id": thread_id}


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
async def conversation_run_status(conversation_id: str) -> RunStatusResponse:
    lg_thread = await get_lg_thread_by_conversation(conversation_id)
    if lg_thread:
        lg_thread_id = lg_thread.get("thread_id", "")
        status = await get_lg_thread_status(lg_thread_id)
    else:
        lg_thread_id = None
        status = "idle"
    return RunStatusResponse(
        thread_id=lg_thread_id or conversation_id,
        run_id=None,
        agent_key=None,
        status=status,
        error=None,
        updated_at=None,
    )


@router.get("/runs/{thread_id}", response_model=RunStatusResponse)
async def run_status(thread_id: str) -> RunStatusResponse:
    status = await get_lg_thread_status(thread_id)
    return RunStatusResponse(
        thread_id=thread_id,
        run_id=None,
        agent_key=None,
        status=status,
        error=None,
        updated_at=None,
    )


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: str) -> Response:
    deleted = await delete_thread_state(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")
    return Response(status_code=204)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str) -> Response:
    deleted = await delete_conversation_state(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")
    return Response(status_code=204)
