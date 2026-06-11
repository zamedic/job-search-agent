"""
Smoke test for the JobScope backend with mocked MiniMax endpoints.

The agent uses httpx.Client to call:
  - POST {chat_base}/v1/messages         (streaming SSE)
  - POST {search_base}/v1/coding_plan/search  (JSON)

We monkey-patch httpx.Client.stream and httpx.Client.post to return
fixtures that simulate the real MiniMax protocol:
  - First chat turn: model emits one tool_use for web_search.
  - Search returns canned results.
  - Second chat turn: model emits a final text response, no tool_use.

Then we assert the SSE events, session persistence, and CRUD endpoints
all work end-to-end.

Run from project root:
    .venv/bin/python -m tests.smoke
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Set fake key + data dir BEFORE importing the app.
os.environ["MINIMAX_API_KEY"] = "sk-fak...test"
os.environ["JOBAGENT_DATA_DIR"] = "/tmp/jobscope-smoke-sessions"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from fastapi.testclient import TestClient  # noqa: E402

import backend.agent as agent_mod
from backend.app import app  # noqa: E402

# ----------------------------------------------------------------------
# Fixture builders
# ----------------------------------------------------------------------

def sse_event(event_type: str, data: dict) -> bytes:
    """Format a single SSE event as bytes."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def make_stream_response(events: list[bytes]) -> httpx.Response:
    """Build a fake streaming Response. httpx.Response accepts a stream
    callable in the `stream` interface; for our purposes we just need
    iter_lines to yield the SSE lines."""
    def stream_iter() -> httpx.SyncByteStream:
        return _ByteStream(events)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_ByteStream(events),
    )


class _ByteStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self):
        for c in self._chunks:
            yield c

    def close(self) -> None:
        pass


def build_chat_stream(events_data: list[tuple[str, dict]]) -> bytes:
    """Turn a list of (event_type, data) tuples into SSE bytes."""
    out = b""
    for et, d in events_data:
        out += f"event: {et}\ndata: {json.dumps(d)}\n\n".encode("utf-8")
    return out


# Two scripted chat turns. Each is a full SSE stream ending in message_stop.
CHAT_TURN_1 = build_chat_stream([
    ("message_start", {
        "type": "message_start",
        "message": {"id": "msg1", "type": "message", "role": "assistant",
                    "content": [], "model": "MiniMax-M3",
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 0}}
    }),
    ("ping", {"type": "ping"}),
    ("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": "Researching the Berlin market... "}
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("content_block_start", {
        "type": "content_block_start", "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_1",
                          "name": "web_search", "input": {}}
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 1,
        "delta": {"type": "input_json_delta",
                  "partial_json": json.dumps({"query": "senior backend engineer salary Berlin 2026"})}
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"input_tokens": 100, "output_tokens": 30}
    }),
    ("message_stop", {"type": "message_stop"}),
])

CHAT_TURN_2 = build_chat_stream([
    ("message_start", {
        "type": "message_start",
        "message": {"id": "msg2", "type": "message", "role": "assistant",
                    "content": [], "model": "MiniMax-M3",
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 200, "output_tokens": 0}}
    }),
    ("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": "# Market Summary\n\n"}
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": "Berlin backend market is **very competitive**."}
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"input_tokens": 200, "output_tokens": 50}
    }),
    ("message_stop", {"type": "message_stop"}),
])

# Phase 3 turn: model does 2 web searches, then emits a candidate profile.
# Scripted: first call uses web_search, second call emits the final text.
CHAT_TURN_3 = build_chat_stream([
    ("message_start", {
        "type": "message_start",
        "message": {"id": "msg3", "type": "message", "role": "assistant",
                    "content": [], "model": "MiniMax-M3",
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 500, "output_tokens": 0}}
    }),
    ("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "tool_use", "id": "toolu_3a",
                          "name": "web_search", "input": {}}
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "input_json_delta",
                  "partial_json": json.dumps({"query": "common reasons senior backend engineer leave current job 2026"})}
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("content_block_start", {
        "type": "content_block_start", "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_3b",
                          "name": "web_search", "input": {}}
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 1,
        "delta": {"type": "input_json_delta",
                  "partial_json": json.dumps({"query": "senior engineer interview motivators 2026 hiring market"})}
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"input_tokens": 500, "output_tokens": 60}
    }),
    ("message_stop", {"type": "message_stop"}),
])

# Phase 3 final turn: model emits the candidate personality profile.
CHAT_TURN_4 = build_chat_stream([
    ("message_start", {
        "type": "message_start",
        "message": {"id": "msg4", "type": "message", "role": "assistant",
                    "content": [], "model": "MiniMax-M3",
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 800, "output_tokens": 0}}
    }),
    ("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta",
                  "text": "## Candidate Personality Profile\n\n"
                          "1. **Stagnation-driven** — senior engineers who have stopped learning...\n"
                          "2. **Comp-blocked** — high performers underpaid at their current role...\n"
                          "3. **Remote-priority** — value location flexibility above comp...\n\n"
                          "## Anti-personas\n- Junior devs (not senior enough)...\n\n"
                          "Want me to proceed to the JDs?"}
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"input_tokens": 800, "output_tokens": 200}
    }),
    ("message_stop", {"type": "message_stop"}),
])

# Phase 4 turn: model emits two target JDs as a final text response.
CHAT_TURN_5 = build_chat_stream([
    ("message_start", {
        "type": "message_start",
        "message": {"id": "msg5", "type": "message", "role": "assistant",
                    "content": [], "model": "MiniMax-M3",
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 1000, "output_tokens": 0}}
    }),
    ("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}
    }),
    ("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta",
                  "text": "## JD 1: Staff Backend Engineer — Platform Reliability\n\nFor the stagnation-driven slice.\n\n...\n\n## JD 2: Senior Backend Engineer — Cloud Native\n\nFor the remote-priority slice.\n\n..."}
    }),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"input_tokens": 1000, "output_tokens": 1500}
    }),
    ("message_stop", {"type": "message_stop"}),
])

SEARCH_RESPONSE = {
    "organic": [
        {"title": "Senior Software Engineer Salary in Berlin",
         "link": "https://www.levels.fyi/t/software-engineer/levels/senior/locations/berlin-deu",
         "snippet": "Average base €84,177 - €120,180.",
         "date": "2026-01-15"},
        {"title": "Glassdoor — Berlin Senior Software Engineer",
         "link": "https://www.glassdoor.com/Salaries/berlin-germany-senior-software-engineer-salary",
         "snippet": "Average $86,250 per year in Berlin, Germany.",
         "date": ""},
    ]
}


# ----------------------------------------------------------------------
# Monkey-patch httpx.Client.stream and httpx.Client.post
# ----------------------------------------------------------------------
class MockState:
    chat_calls: list[dict] = []
    search_calls: list[dict] = []
    # _chat_iter is consumed by the agent loop. Each iter of the loop
    # consumes one entry. The model emits 2 turns per phase that uses
    # tools (tool_use iter + end_turn iter), and 1 turn for a final
    # text-only phase.
    _chat_iter = iter([
        CHAT_TURN_1,  # Phase 2: tool_use
        CHAT_TURN_2,  # Phase 2: end_turn (report)
        CHAT_TURN_3,  # Phase 3: tool_use
        CHAT_TURN_4,  # Phase 3: end_turn (profile)
        CHAT_TURN_5,  # Phase 4: end_turn (JDs)
    ])


_orig_stream = httpx.Client.stream
_orig_post = httpx.Client.post


class _FakeStreamCtx:
    """Mimics httpx's Client.stream() context manager.

    httpx.Client.stream() returns an object whose __enter__ gives back
    a Response. We do the same: build a Response from a byte stream,
    and wrap it in this context manager so the `with` statement works.
    """
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def __enter__(self) -> httpx.Response:
        return self._response

    def __exit__(self, *exc_info) -> bool:
        self._response.close()
        return False


def fake_stream(self, method, url, **kwargs):
    """Return a context manager yielding a Response with our SSE bytes."""
    if "/v1/messages" in str(url):
        MockState.chat_calls.append({"url": str(url), "body": kwargs.get("json")})
        try:
            next_bytes = next(MockState._chat_iter)
        except StopIteration:
            next_bytes = b""
        # Build a Request so the Response has one and raise_for_status works.
        request = httpx.Request(
            "POST", str(url),
            headers={"x-api-key": "x", "anthropic-version": "2023-06-01"},
            json=kwargs.get("json"),
        )
        resp = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ByteStream([next_bytes]),
            request=request,
        )
        return _FakeStreamCtx(resp)
    return _orig_stream(self, method, url, **kwargs)


def fake_post(self, url, **kwargs):
    if "/v1/coding_plan/search" in str(url):
        MockState.search_calls.append({"url": str(url), "body": kwargs.get("json")})
        # Build a Request so the Response has one and raise_for_status works.
        request = httpx.Request(
            "POST", str(url),
            headers={"Authorization": "Bearer x", "Content-Type": "application/json"},
            json=kwargs.get("json"),
        )
        return httpx.Response(
            200, json=SEARCH_RESPONSE,
            headers={"content-type": "application/json"},
            request=request,
        )
    return _orig_post(self, url, **kwargs)


# Patch
httpx.Client.stream = fake_stream
httpx.Client.post = fake_post


# ----------------------------------------------------------------------
# Wipe data dir before we start.
# ----------------------------------------------------------------------
data_dir = Path("/tmp/jobscope-smoke-sessions")
if data_dir.exists():
    shutil.rmtree(data_dir)
data_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Run the tests
# ----------------------------------------------------------------------
def main():
    client = TestClient(app)

    print("=== /api/health ===")
    r = client.get("/api/health")
    print(f"  status={r.status_code} body={r.json()}")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["model"] == "MiniMax-M3"
    assert r.json()["has_api_key"] is True

    print("=== POST /api/sessions ===")
    r = client.post("/api/sessions", json={})
    session_id = r.json()["id"]
    print(f"  status={r.status_code} id={session_id}")
    assert r.status_code == 200

    print("=== POST /api/chat (streamed) ===")
    received = []
    with client.stream("POST", "/api/chat",
                       json={"session_id": session_id,
                             "message": "I need to hire a senior backend engineer in Berlin."}) as r:
        assert r.status_code == 200, f"chat failed: {r.status_code}"
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                step = json.loads(payload)
            except json.JSONDecodeError:
                continue
            received.append(step)
            t = step.get("type", "?")
            if t == "text_delta":
                print(f"  [{t}] {step.get('text','')!r}")
            elif t == "search_start":
                print(f"  [{t}] query={step.get('query')!r}")
            elif t == "search_done":
                print(f"  [{t}] query={step.get('query')!r}")
            elif t == "message_done":
                print(f"  [{t}] text={step.get('text','')[:60]!r}...")
            elif t == "error":
                print(f"  [{t}] {step.get('message')!r}")
            elif t == "end":
                print(f"  [{t}]")
            else:
                print(f"  [{t}] {step}")

    types = [s["type"] for s in received]
    print(f"\nReceived {len(received)} events. Types: {types}")
    assert "search_start" in types
    assert "search_done" in types
    assert "text_delta" in types
    assert "message_done" in types
    assert "end" in types
    print("  ✓ all expected event types received")

    # The model called the chat endpoint twice (one with tool_use, one without).
    print(f"\nChat endpoint was called {len(MockState.chat_calls)} times")
    assert len(MockState.chat_calls) == 2
    # The search endpoint was called once.
    print(f"Search endpoint was called {len(MockState.search_calls)} times")
    assert len(MockState.search_calls) == 1
    assert MockState.search_calls[0]["body"] == {"q": "senior backend engineer salary Berlin 2026"}
    print("  ✓ chat and search endpoints called correctly")

    # The second chat call should carry the tool_use + tool_result in history.
    iter2 = MockState.chat_calls[1]["body"]
    iter2_messages = iter2["messages"]
    has_tool_use = any(
        b.get("type") == "tool_use"
        for m in iter2_messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
    )
    has_tool_result = any(
        b.get("type") == "tool_result"
        for m in iter2_messages
        for b in (m["content"] if isinstance(m["content"], list) else [])
    )
    assert has_tool_use, "iter 2 history missing tool_use block"
    assert has_tool_result, "iter 2 history missing tool_result block"
    print("  ✓ iter 2 message history carries tool_use + tool_result blocks")

    # Multi-turn flow: user says "yes, build the profile", agent does
    # Phase 3 (2 searches, then final profile). Then user says "proceed
    # to JDs", agent does Phase 4 (final text, no searches).
    print("\n=== Second turn: user confirms, agent does Phase 3 (profile) ===")
    received = []
    with client.stream("POST", "/api/chat",
                       json={"session_id": session_id,
                             "message": "Yes, build the candidate personality profile."}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                step = json.loads(payload)
            except json.JSONDecodeError:
                continue
            received.append(step)
            t = step.get("type")
            if t == "search_start":
                print(f"  [search] {step.get('query')!r}")
            elif t == "message_done":
                print(f"  [done] {len(step.get('text',''))} chars")
            elif t == "error":
                print(f"  [ERROR] {step.get('message')!r}")
    types = [s["type"] for s in received]
    # Phase 3 should produce 2 search_start + 2 search_done + text_deltas + message_done
    assert types.count("search_start") == 2, f"expected 2 searches in Phase 3, got {types.count('search_start')}"
    assert "message_done" in types
    print(f"  ✓ Phase 3 ran (2 searches + profile emitted)")

    print("\n=== Third turn: user approves profile, agent does Phase 4 (JDs) ===")
    received = []
    with client.stream("POST", "/api/chat",
                       json={"session_id": session_id,
                             "message": "Looks good. Now write the targeted job descriptions."}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                step = json.loads(payload)
            except json.JSONDecodeError:
                continue
            received.append(step)
            t = step.get("type")
            if t == "search_start":
                print(f"  [search] {step.get('query')!r}")
            elif t == "message_done":
                print(f"  [done] {len(step.get('text',''))} chars")
            elif t == "error":
                print(f"  [ERROR] {step.get('message')!r}")
    types = [s["type"] for s in received]
    # Phase 4 should be a single text response, no searches
    assert "search_start" not in types, f"Phase 4 should not search, got: {types}"
    assert "message_done" in types
    final_jd = next(s for s in received if s["type"] == "message_done")
    assert "JD" in final_jd.get("text", "") or "job" in final_jd.get("text", "").lower()
    print(f"  ✓ Phase 4 ran (JDs emitted, no extra searches)")

    # The chat endpoint should have been called 5 times total: 2 each for
    # Phase 2 and Phase 3, 1 for Phase 4.
    print(f"\nTotal chat calls: {len(MockState.chat_calls)}")
    assert len(MockState.chat_calls) == 5, f"expected 5 chat calls, got {len(MockState.chat_calls)}"
    # And the search endpoint was called 3 times: 1 in Phase 2, 2 in Phase 3.
    assert len(MockState.search_calls) == 3, f"expected 3 search calls, got {len(MockState.search_calls)}"
    print(f"  ✓ 5 chat calls, 3 search calls (correct distribution)")

    # Session was persisted.
    r = client.get(f"/api/sessions/{session_id}")
    s = r.json()
    print(f"\nSession has {len(s['messages'])} messages")
    for m in s["messages"]:
        preview = m["content"][:60].replace("\n", " ")
        print(f"  [{m['role']}] {preview}{'...' if len(m['content']) > 60 else ''}")
    # 3 user turns + 3 assistant turns = 6 messages after the full flow
    assert len(s["messages"]) == 6
    assert s["messages"][0]["role"] == "user"
    assert s["messages"][1]["role"] == "assistant"
    assert "Berlin" in s["messages"][1]["content"]
    assert "competitive" in s["messages"][1]["content"]
    # Last assistant message is the JDs
    assert s["messages"][-1]["role"] == "assistant"
    assert "JD" in s["messages"][-1]["content"] or "job" in s["messages"][-1]["content"].lower()
    print("  ✓ session persisted with full multi-turn history")

    # Rename and delete.
    print("\n=== PATCH /api/sessions/{id} (rename) ===")
    r = client.patch(f"/api/sessions/{session_id}", json={"title": "Berlin backend search"})
    assert r.status_code == 200
    assert r.json()["title"] == "Berlin backend search"
    print("  ✓ rename works")

    print("\n=== DELETE /api/sessions/{id} ===")
    r = client.delete(f"/api/sessions/{session_id}")
    assert r.json()["deleted"] is True
    assert client.get(f"/api/sessions/{session_id}").status_code == 404
    print("  ✓ delete works (subsequent GET 404)")

    # Also: web_search() direct unit test.
    print("\n=== web_search() direct call (returns packed text) ===")
    text = agent_mod.web_search("python developer", api_key="sk-fak")
    assert "Search results for: python developer" in text
    assert "levels.fyi" in text
    print("  ✓ web_search returns packed text block")
    print(text[:200])

    print("\n=== All smoke tests passed ===")


if __name__ == "__main__":
    main()
