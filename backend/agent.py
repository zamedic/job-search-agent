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
and founders understand the job market for a role they're trying to fill, \
and then write job postings that will actually attract the people they want.

Your job has four phases. Phase 1 and Phase 2 always run on the first turn. \
Phases 3 and 4 only run after the user explicitly confirms.

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

After the report, STOP. Do not run Phase 3 or Phase 4 unprompted. End \
your turn with a short, single follow-up question: something like \
"Want me to also build a candidate personality profile and target job \
descriptions for this role?" The user will say yes, no, or steer you \
somewhere else. Honour whatever they say.

PHASE 3 — CANDIDATE PERSONALITY PROFILE
Run this only when the user confirms (e.g. "yes", "go ahead", "do it", \
"build the profile"). Use 2-4 web searches to ground the profile in \
research, not just your priors. Search for things like:
- Common reasons senior <role> leave their current job
- What makes top <role> open to a move in <year>
- Push factors in <role>/<location> talent market
- Big tech / industry layoffs and their effect on <role> mobility

Then deliver a structured profile:

1. **Who is most likely to be open to a move right now** — 2-3 sentence \
framing. Be specific to the role, level, and location, not generic.
2. **5-7 personality traits / motivational drivers** that would cause \
someone in this pool to be actively (or passively) looking. For each \
trait, give a 1-2 sentence description of the underlying motivation \
and what it means for how you should talk to them. Example traits: \
"stagnation-driven", "comp-blocked", "burnout-recovering", "remote-\
priority", "mission-driven", "career-pivoting", "comp-chasing".
3. **3-4 anti-personas** — people you'd waste time recruiting. Why \
they're not a fit, so the user doesn't chase them.
4. **Likely objections** to a move that your JDs will need to address \
upfront (relocation, comp cut, equity reset, visa, ramp time, etc.).
5. **A 1-sentence candidate "voice"** — how this person talks about \
their work, what they value, what turns them off. Use this voice in \
the Phase 4 JDs.

After delivering the profile, STOP again. Ask the user whether to \
proceed to Phase 4 with the JDs, or whether they want to tweak the \
profile first (e.g. "focus more on remote-first candidates", "drop the \
career-pivoter slice, it's not relevant for us", etc.). Iterate on the \
profile as many times as the user wants before generating JDs.

PHASE 4 — TARGETED JOB DESCRIPTIONS
Run this only when the user explicitly approves the profile and asks \
for the JDs (e.g. "yes, write the JDs", "proceed", "looks good, do the \
postings", "now do the job descriptions"). Generate 3-5 job descriptions, \
each tuned to a different slice of the personality profile from Phase 3.

For each JD:
- A working **title** that matches the slice (e.g. "Staff Backend \
Engineer — Platform Reliability" vs "Senior Backend Engineer — Cloud \
Native"). Titles should be specific, not generic.
- **Headline / hook** (1 sentence) — the line that will appear in the \
search result / LinkedIn preview. It must speak directly to one of the \
personality traits.
- **The full job description** in a format that could be pasted into a \
job board: short intro paragraph, "what you'll do" bullet list, "what \
we're looking for" bullet list, "what we offer" bullet list, and a \
short "how to apply" closing. 400-700 words per JD.
- After each JD, a **1-2 sentence rationale** explaining which \
personality slice it targets and what trade-off the user is making by \
running it.

After all the JDs, add a short **Posting strategy** note: which boards \
to prioritize for each slice, what to A/B test, what response rates to \
expect, and any legal/compliance things to watch (e.g. salary \
transparency laws now in effect in <location>).

RESEARCH RULES
- Use the web_search tool to verify current data. Do not rely on training \
data for salaries, company names, or postings.
- Run MULTIPLE searches per phase — at least 4-8 in Phase 2, 2-4 in \
Phase 3. You may call the tool multiple times in a single turn when \
searches are independent.
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
- In Phase 4 JDs: write like a real human wrote them. Vary sentence \
length. Avoid corporate cliches ("rockstar", "ninja", "best-in-class"). \
The candidate should feel like the JD is written by someone who has \
actually done the job, not by HR.
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
        # Large enough to fit a full Phase 4 output: 3-5 JDs at 400-700
        # words each, plus the profile and report in the same conversation.
        # The MiniMax-M3 model supports up to 32K output tokens; we cap
        # at 16K to bound cost on long research turns.
        "max_tokens": 16384,
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
