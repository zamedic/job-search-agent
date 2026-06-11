"""
Thread-safe JSON file storage for chat sessions.

Good enough for an internal team tool. One file per session, written
atomically. Replace with Redis/Postgres if you need real concurrency
or persistence beyond a single node.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("JOBAGENT_DATA_DIR", "data/sessions"))


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# In-process lock so two concurrent requests don't clobber the same file.
# (Cross-process / cross-node is not handled — add a real DB if needed.)
_write_lock = threading.Lock()


def list_sessions() -> list[dict]:
    """Return metadata for all sessions, newest first."""
    _ensure_data_dir()
    out: list[dict] = []
    for path in DATA_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append({
                "id": data["id"],
                "title": data.get("title", "(untitled)"),
                "created_at": data.get("created_at", 0),
                "updated_at": data.get("updated_at", 0),
                "message_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            log.warning("skipping corrupt session file %s: %s", path, exc)
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out


def get_session(session_id: str) -> dict | None:
    path = DATA_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read session %s: %s", session_id, exc)
        return None


def create_session(title: str | None = None) -> dict:
    _ensure_data_dir()
    now = time.time()
    session = {
        "id": uuid.uuid4().hex,
        "title": title or "New research",
        "created_at": now,
        "updated_at": now,
        "messages": [],  # list of {role, content, ts}  — only user/assistant text
    }
    _save(session)
    return session


def append_message(session_id: str, role: str, content: str) -> None:
    """Append a user/assistant text turn to a session."""
    with _write_lock:
        session = get_session(session_id)
        if session is None:
            raise KeyError(f"session {session_id} not found")
        session["messages"].append({
            "role": role,
            "content": content,
            "ts": time.time(),
        })
        session["updated_at"] = time.time()
        # Auto-title from the first user message if still generic.
        if session["title"] in ("New research", "(untitled)") and role == "user":
            session["title"] = content.strip().splitlines()[0][:80] or "New research"
        _save(session)


def rename_session(session_id: str, title: str) -> bool:
    with _write_lock:
        session = get_session(session_id)
        if session is None:
            return False
        session["title"] = title
        session["updated_at"] = time.time()
        _save(session)
        return True


def delete_session(session_id: str) -> bool:
    with _write_lock:
        path = DATA_DIR / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False


def _save(session: dict) -> None:
    """Write atomically: write to temp file, then rename."""
    path = DATA_DIR / f"{session['id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# Useful for tests / batch tools.
def iter_all_sessions() -> Iterator[dict]:
    for meta in list_sessions():
        s = get_session(meta["id"])
        if s is not None:
            yield s
