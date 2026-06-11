"""
Job Market Research Agent — talks to MiniMax directly via httpx.

Two MiniMax endpoints are used:
  - Chat:    POST {MINIMAX_CHAT_BASE}/v1/messages
             This is MiniMax's Anthropic-compatible endpoint
             (https://api.minimax.io/anthropic by default), so we use
             the standard x-api-key auth and the Anthropic SSE event
             format.
  - Search:  POST {MINIMAX_SEARCH_BASE}/v1/coding_plan/search
             This is a MiniMax-native endpoint, not under the
             /anthropic prefix, and uses standard Bearer auth.

The model is given a custom `web_search` function tool. When it emits
tool_use blocks, we hit the search endpoint and feed results back as
tool_result blocks, letting the model iterate until it's satisfied
and emits a final assistant message.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from typing import Iterator

import httpx

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Config — read from env, with sensible defaults for MiniMax.
# ----------------------------------------------------------------------
# Chat is served through MiniMax's Anthropic-compatible endpoint so
# the same x-api-key + SSE protocol works. Search is a separate
# MiniMax-native endpoint (different URL prefix, different auth style).
DEFAULT_CHAT_BASE = os.environ.get("MINIMAX_CHAT_BASE", "https://api.minimax.io/anthropic")
DEFAULT_SEARCH_BASE = os.environ.get("MINIMAX_SEARCH_BASE", "https://api.minimax.io")
DEFAULT_MODEL = os.environ.get("JOB_MODEL", "MiniMax-M3")
ANTHROPIC_VERSION = "2023-06-01"

# Max tool uses per turn — caps cost if the model goes in a loop.
MAX_SEARCHES_PER_TURN = 10

# Max agentic loop iterations — safety net so we never hang.
MAX_LOOP_ITERATIONS = 8

# ----------------------------------------------------------------------
# System prompt — defines the agent's job, voice, and report structure.
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are JobScope, a research agent that helps hiring managers, recruiters, \
and founders understand the job market for a role they're trying to fill.

Your job has two phases:

PHASE 1 — CLARIFY (only when needed)
If the user's request is vague (e.g. "I need to hire a dev in Berlin"), \
ask 2-4 sharp clarifying questions in a SINGLE message. Cover: the exact \
role title, location and remote/hybrid preferences, seniority, must-have \
skills, employment type, and timeline. Skip questions the user has \
already answered. Do NOT begin researching in this phase.

PHASE 2 — RESEARCH AND REPORT
Once the brief is clear, conduct deep web research and deliver a \
structured report. The report MUST contain these sections, in order:

1. **Market Summary** — 2-3 sentence overview of the market.
2. **Talent Availability** — Easy / Moderate / Hard / Very Hard, with a \
short justification based on supply indicators you found.
3. **Demand Signals** — Hiring volume, growth trends, time-to-fill, \
specific companies hiring.
4. **Salary Range** — Provide a base salary range AND a total comp range \
where relevant. Always cite the source URL and the date of the data. \
Give ranges in the local currency, with a USD/EUR/GBP equivalent in \
parentheses. Distinguish by seniority level (junior / mid / senior / \
staff+).
5. **Sample Job Postings** — 3-5 real, currently-live job postings with \
the company name, role title, location, salary if posted, and a direct \
URL. Include a 1-2 sentence summary of what makes each one interesting \
or representative.
6. **Recommendations** — 3-5 actionable suggestions for the user (e.g. \
"sponsor a visa", "consider contract-to-hire", "expect 8-12 weeks to fill").
7. **Caveats** — Be honest about limitations of the data (small sample \
size, stale postings, geographic edge cases, etc.).

RESEARCH RULES
- Use the web_search tool to verify current data. Do not rely on training \
data for salaries, company names, or postings.
- Run MULTIPLE searches — at least 4-8 per research task — covering: \
salary data, job board listings, industry/talent reports, company career \
pages, and recent news. You may call the tool multiple times in a single \
turn when searches are independent.
- Prefer authoritative sources: Glassdoor, Levels.fyi, Payscale, \
Indeed, LinkedIn, company career pages, government labor stats, \
Stack Overflow Developer Survey, Robert Half salary guide, Hays, \
Michael Page, etc.
- When citing salary data, ALWAYS include the source URL and the date \
the data was published or last updated.
- When listing job postings, include the direct URL to the posting. \
If a posting has expired or you cannot verify it, do not include it.
- Be honest about what you could not find. If data is thin for a \
specific geography or role, say so.

VOICE AND FORMAT
- Use clean Markdown. Headers, bullet points, tables where they help.
- Be concise. The user is a busy hiring manager, not a reader of essays.
- No hype, no filler phrases like "in today's competitive market". \
Just the facts and your synthesis.
- Use the local currency of the role's location as the primary unit, \
with a parenthetical conversion to USD or EUR if helpful.
"""

# ----------------------------------------------------------------------
# Tool schema (Anthropic-compatible function tool).
# ----------------------------------------------------------------------
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current information. Returns titles, links, "
        "and short snippets from the top results. Use this to look up "
        "salary data, job postings, company info, and recent market news."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            }
        },
        "required": ["query"],
    },
}


# ----------------------------------------------------------------------
# Step event types streamed to the caller.
# ----------------------------------------------------------------------
# Step dict shapes:
#   {"type": "search_start", "query": "..."}     — search starting
#   {"type": "search_done",  "query": "..."}     — search returned
#   {"type": "text_delta",   "text": "..."}      — model typing
#   {"type": "message_done", "text": "..."}      — final assistant message
#   {"type": "error",        "message": "..."}   — fatal error
Step = dict


# ----------------------------------------------------------------------
# Search helper — calls MiniMax's coding_plan/search endpoint.
# ----------------------------------------------------------------------
def web_search(
    query: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_SEARCH_BASE,
    client: httpx.Client | None = None,
) -> str:
    """Run a web search and return a text summary of the top results.

    We pack the JSON results into a compact text block so the model
    can read them. The tool_result content is sent as a plain string.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=30)

    try:
        resp = client.post(
            f"{base_url}/v1/coding_plan/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"q": query},
        )
        resp.raise_for_status()
        data = resp.json()
    finally:
        if owns_client:
            client.close()

    organic = data.get("organic") or []
    if not organic:
        return f"No results for query: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(organic, 1):
        title = r.get("title", "(no title)")
        link = r.get("link", "")
        snippet = r.get("snippet", "")
        date = r.get("date", "")
        lines.append(f"[{i}] {title}")
        if date:
            lines.append(f"    Date: {date}")
        if link:
            lines.append(f"    URL: {link}")
        if snippet:
            lines.append(f"    {snippet}")
        lines.append("")
    return "\n".join(lines).strip()


# ----------------------------------------------------------------------
# SSE parser — yields one event dict per SSE message.
# ----------------------------------------------------------------------
def _iter_sse_events(lines: Iterator[str]) -> Iterator[dict]:
    """Parse SSE `data: {...}` lines into dicts.

    Anthropic-style SSE has the form:
        event: message_start
        data: {"type":"message_start", ...}

        event: content_block_delta
        data: {"type":"content_block_delta", ...}

    We ignore the `event:` line and parse `data:` payloads only.
    """
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            log.warning("bad SSE payload: %r", payload[:120])


# ----------------------------------------------------------------------
# Chat helper — POSTs to /v1/messages and yields step events.
# Also returns the final assistant content blocks (so the caller can
# push them back into the message history).
# ----------------------------------------------------------------------
def _stream_chat(
    *,
    api_key: str,
    chat_base: str,
    model: str,
    messages: list[dict],
    on_step,
    http_client: httpx.Client,
) -> tuple[list[dict], str | None]:
    """Run one chat turn. Returns (assistant_content_blocks, stop_reason).

    `assistant_content_blocks` is a list of dicts ready to be pushed
    back into the messages history as an assistant turn. The caller
    does not need to know the streaming internals.
    """
    body = {
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "tools": [WEB_SEARCH_TOOL],
        "messages": messages,
        "stream": True,
    }

    # State for the current stream — one response can contain many
    # content blocks, each with their own index. We accumulate them
    # so the final assistant turn is well-formed.
    blocks: dict[int, dict] = {}        # index -> in-progress block dict
    stop_reason: str | None = None

    with http_client.stream(
        "POST",
        f"{chat_base}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        json=body,
    ) as resp:
        resp.raise_for_status()

        for event in _iter_sse_events(resp.iter_lines()):
            t = event.get("type")

            if t == "content_block_start":
                idx = event["index"]
                block = dict(event["content_block"])  # shallow copy
                # Normalize: text blocks need an empty text field,
                # tool_use blocks need an empty input dict to be filled.
                if block.get("type") == "text":
                    block.setdefault("text", "")
                elif block.get("type") == "tool_use":
                    block.setdefault("input", {})
                    block["input"] = dict(block.get("input") or {})
                    # Track the query for search_start events
                    blocks[idx] = block
                    name = block.get("name", "")
                    if name == "web_search":
                        # Will be updated by deltas; emit search_start later.
                        pass
                    continue
                blocks[idx] = block

            elif t == "content_block_delta":
                idx = event["index"]
                delta = event.get("delta", {})
                dt = delta.get("type")
                block = blocks.get(idx)
                if not block:
                    continue
                if dt == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        block["text"] = block.get("text", "") + text
                        on_step({"type": "text_delta", "text": text})
                elif dt == "input_json_delta":
                    # Concatenate partial JSON, parse at block_stop.
                    block["input_raw"] = block.get("input_raw", "") + delta.get("partial_json", "")
                elif dt == "thinking_delta":
                    # We don't render thinking to the UI; just collect.
                    block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")

            elif t == "content_block_stop":
                idx = event["index"]
                block = blocks.get(idx)
                if not block:
                    continue
                if block.get("type") == "tool_use":
                    # Finalize input.
                    raw = block.pop("input_raw", "")
                    if raw:
                        try:
                            block["input"] = json.loads(raw)
                        except json.JSONDecodeError:
                            log.warning("bad tool input JSON: %r", raw[:120])
                            block["input"] = {}
                    if block.get("name") == "web_search":
                        q = (block.get("input") or {}).get("query", "")
                        if q:
                            on_step({"type": "search_start", "query": q})

            elif t == "message_delta":
                stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason

            elif t == "error":
                err = event.get("error") or {}
                raise RuntimeError(f"stream error: {err.get('type','?')}: {err.get('message','?')}")

            # message_start, message_stop, ping: ignore

    # Materialize blocks into a list, in order of index.
    assistant_content: list[dict] = []
    for idx in sorted(blocks.keys()):
        b = blocks[idx]
        # Drop the temporary input_raw field, keep everything else.
        b.pop("input_raw", None)
        assistant_content.append(b)

    return assistant_content, stop_reason


# ----------------------------------------------------------------------
# The agent loop.
# ----------------------------------------------------------------------
def _run_blocking(
    *,
    api_key: str,
    chat_base: str,
    search_base: str,
    model: str,
    messages: list[dict],
    q: "queue.Queue[Step | None]",
) -> None:
    """Run the agent loop. Pushes step events to q; ends with q.put(None)."""
    def emit(step: Step) -> None:
        q.put(step)

    with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0)) as http:
        try:
            for iteration in range(MAX_LOOP_ITERATIONS):
                log.info("agent loop iteration %d (messages=%d)", iteration, len(messages))

                assistant_content, stop_reason = _stream_chat(
                    api_key=api_key,
                    chat_base=chat_base,
                    model=model,
                    messages=messages,
                    on_step=emit,
                    http_client=http,
                )

                if not assistant_content:
                    emit({"type": "error", "message": "Empty response from model"})
                    return

                # Push assistant turn into history.
                messages.append({"role": "assistant", "content": assistant_content})

                if stop_reason != "tool_use":
                    # Final turn — join the text blocks for the UI and persist.
                    final_text = "\n".join(
                        b.get("text", "")
                        for b in assistant_content
                        if b.get("type") == "text"
                    ).strip()
                    emit({"type": "message_done", "text": final_text})
                    return

                # Otherwise the model called our tool. Run the searches
                # in parallel (faster, model often requests multiple).
                tool_uses = [b for b in assistant_content if b.get("type") == "tool_use"]
                if not tool_uses:
                    emit({"type": "error", "message": "stop_reason=tool_use but no tool_use blocks"})
                    return

                tool_results: list[dict] = []
                threads: list[threading.Thread] = []
                results_lock = threading.Lock()

                def run_one(block: dict) -> None:
                    name = block.get("name", "")
                    inp = block.get("input") or {}
                    tid = block.get("id", "")
                    if name != "web_search":
                        result_text = f"Tool '{name}' is not supported."
                    else:
                        query = inp.get("query", "")
                        try:
                            result_text = web_search(
                                query,
                                api_key=api_key,
                                base_url=search_base,
                                client=http,
                            )
                            emit({"type": "search_done", "query": query})
                        except Exception as exc:  # noqa: BLE001
                            result_text = f"Search error: {type(exc).__name__}: {exc}"
                            emit({"type": "search_done", "query": query})

                    # Build the tool_result block. Truncate to keep
                    # tokens bounded — the model only needs the gist.
                    if len(result_text) > 12_000:
                        result_text = result_text[:12_000] + "\n... [truncated]"

                    with results_lock:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": result_text,
                        })

                for tu in tool_uses:
                    t = threading.Thread(target=run_one, args=(tu,), daemon=True)
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()

                # Push tool_result blocks back as a single user turn.
                messages.append({"role": "user", "content": tool_results})

            emit({"type": "error", "message": f"Hit max loop iterations ({MAX_LOOP_ITERATIONS})"})

        except Exception as exc:  # noqa: BLE001
            log.exception("agent loop failed")
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            q.put(None)


def run_research(
    *,
    api_key: str,
    chat_base: str,
    search_base: str,
    model: str,
    messages: list[dict],
) -> Iterator[Step]:
    """
    Run a research turn, yielding Step dicts as the agent works.

    The caller is expected to consume this generator. The messages list
    is mutated in-place with each assistant turn (and tool_result turn),
    so the caller can keep the same list for the next turn.
    """
    q: "queue.Queue[Step | None]" = queue.Queue()

    def worker() -> None:
        _run_blocking(
            api_key=api_key,
            chat_base=chat_base,
            search_base=search_base,
            model=model,
            messages=messages,
            q=q,
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        step = q.get()
        if step is None:
            break
        yield step
