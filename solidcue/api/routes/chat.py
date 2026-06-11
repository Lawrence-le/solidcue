"""User-facing routed chat endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from solidcue.api.schemas import StreamChatRequest
from solidcue.services.run_engine import stream_router_chat_events
from solidcue.user.loader import load_user_profile

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse_frame(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/stream")
async def stream(request: StreamChatRequest) -> StreamingResponse:
    if request.user_input is None and request.resume_value is None:
        raise HTTPException(
            status_code=400,
            detail="user_input or resume_value is required",
        )
    router_provider = request.router_provider
    if request.user_input is not None and router_provider is None:
        router_provider = load_user_profile().router_provider
    if request.user_input is not None and router_provider is None:
        raise HTTPException(status_code=400, detail="router provider is not configured")
    if request.resume_value is not None and not request.thread_id and not request.conversation_id:
        raise HTTPException(
            status_code=400,
            detail="thread_id or conversation_id is required to resume",
        )

    async def _generator():
        async for event in stream_router_chat_events(
            thread_id=request.thread_id,
            conversation_id=request.conversation_id,
            user_input=request.user_input,
            resume_value=request.resume_value,
            router_provider_config=router_provider,
        ):
            name = event.get("event")
            data = event.get("data")
            if isinstance(name, str) and isinstance(data, dict):
                yield _sse_frame(name, data)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
