"""
JobScope — web-based job market research agent.

A small FastAPI app that:
  - Serves a single-page chat UI (vanilla HTML/JS) at /
  - Manages chat sessions stored as JSON files (data/sessions/*.json)
  - Streams research progress to the browser via Server-Sent Events
  - Talks to MiniMax (chat via the Anthropic-compatible endpoint,
    web search via the /v1/coding_plan/search endpoint)

Run:
    uvicorn backend.app:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import sessions
from .agent import (
    DEFAULT_CHAT_BASE,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_BASE,
    run_research,
)

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("jobscope")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
# MiniMax uses a single API key for both chat and search. The chat
# endpoint is Anthropic-compatible (x-api-key), the search endpoint
# is a standard Bearer-token API.
API_KEY = os.environ.get("MINIMAX_API_KEY", "").strip()
if not API_KEY:
    # Fall back to the Anthropic-style env var so people with existing
    # configs don't have to rename anything.
    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

CHAT_BASE = os.environ.get("MINIMAX_CHAT_BASE", DEFAULT_CHAT_BASE)
SEARCH_BASE = os.environ.get("MINIMAX_SEARCH_BASE", DEFAULT_SEARCH_BASE)
MODEL = os.environ.get("JOB_MODEL", DEFAULT_MODEL)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if not API_KEY:
    log.warning(
        "MINIMAX_API_KEY (or ANTHROPIC_API_KEY) is not set. The /api/chat "
        "endpoint will fail until it is. Set it in .env or in your environment."
    )

log.info("model=%s chat_base=%s search_base=%s", MODEL, CHAT_BASE, SEARCH_BASE)

# ----------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------
app = FastAPI(title="JobScope — Job Market Research Agent", version="0.2.0")


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session to append this turn to.")
    message: str = Field(..., min_length=1, description="User message text.")


class CreateSessionRequest(BaseModel):
    title: str | None = None


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ----------------------------------------------------------------------
# Session endpoints
# ----------------------------------------------------------------------
@app.get("/api/sessions")
def api_list_sessions() -> list[dict]:
    return sessions.list_sessions()


@app.post("/api/sessions")
def api_create_session(req: CreateSessionRequest | None = None) -> dict:
    title = req.title if req else None
    return sessions.create_session(title=title)


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str) -> dict:
    s = sessions.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s


@app.patch("/api/sessions/{session_id}")
def api_rename_session(session_id: str, req: RenameRequest) -> dict:
    if not sessions.rename_session(session_id, req.title):
        raise HTTPException(status_code=404, detail="session not found")
    return sessions.get_session(session_id)  # type: ignore[return-value]


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str) -> dict:
    deleted = sessions.delete_session(session_id)
    return {"deleted": deleted}


# ----------------------------------------------------------------------
# Chat (SSE streaming)
# ----------------------------------------------------------------------
@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> StreamingResponse:
    """
    Stream research progress for one turn as Server-Sent Events.

    Event types:
      - search_start: {type, query}
      - search_done:  {type, query}
      - text_delta:   {type, text}
      - message_done: {type, text}
      - error:        {type, message}
      - end:          terminal marker
    """
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MINIMAX_API_KEY not configured on the server. Set it in .env.",
        )

    session = sessions.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    # Persist the user's message immediately so a mid-stream crash still
    # keeps the user's question on disk.
    sessions.append_message(req.session_id, "user", req.message)

    # Build the message history from the persisted turns. We pass
    # role/content; system prompt is set in the agent.
    history: list[dict] = [
        {"role": m["role"], "content": m["content"]}
        for m in session["messages"]
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
    # The message we just appended is the last user turn.
    # (Defensive: ensure last is user.)
    if not history or history[-1]["role"] != "user" or history[-1]["content"] != req.message:
        history.append({"role": "user", "content": req.message})

    async def event_stream() -> AsyncIterator[bytes]:
        import asyncio
        import queue as _queue
        import threading

        # The agent runs in a thread; we drain the queue from the event
        # loop. run_in_executor lets the client disconnect cleanly.
        q: "_queue.Queue[dict | None]" = _queue.Queue()

        def step_producer() -> None:
            try:
                for step in run_research(
                    api_key=API_KEY,
                    chat_base=CHAT_BASE,
                    search_base=SEARCH_BASE,
                    model=MODEL,
                    messages=history,
                ):
                    q.put(step)
            except Exception as exc:  # noqa: BLE001
                log.exception("agent thread crashed")
                q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                q.put(None)

        threading.Thread(target=step_producer, daemon=True).start()

        final_text_parts: list[str] = []
        errored = False
        loop = asyncio.get_event_loop()

        try:
            while True:
                step = await loop.run_in_executor(None, q.get)
                if step is None:
                    break

                if step.get("type") == "text_delta":
                    final_text_parts.append(step.get("text", ""))
                elif step.get("type") == "message_done":
                    # message_done carries the authoritative final text.
                    # Reset our buffer to match it.
                    final_text_parts = [step.get("text", "")]
                elif step.get("type") == "error":
                    errored = True
                    sessions.append_message(
                        req.session_id,
                        "assistant",
                        f"_(error: {step.get('message', 'unknown')})_",
                    )

                payload = f"data: {json.dumps(step, ensure_ascii=False)}\n\n"
                yield payload.encode("utf-8")
        finally:
            # Persist the assistant turn if we got a final text.
            try:
                final_text = "".join(final_text_parts).strip()
                if final_text and not errored:
                    sessions.append_message(req.session_id, "assistant", final_text)
            except Exception:  # noqa: BLE001
                log.exception("failed to persist assistant message")
            yield b"data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL,
        "has_api_key": bool(API_KEY),
    }


# ----------------------------------------------------------------------
# Static frontend
# ----------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount the rest of the frontend (CSS, JS) as static files.
# Mounted last so it doesn't shadow the API or index route.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
