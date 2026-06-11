# JobScope — Job Market Research Agent

A small web app that turns a free-text hiring question into a structured
job-market report. Powered by **MiniMax-M3** (via the Anthropic-compatible
chat endpoint) plus a custom `web_search` function tool that hits
MiniMax's `coding_plan/search` endpoint.

The user describes a role they're trying to fill (e.g. "I need a senior
backend engineer in Berlin, hybrid, Go and Kubernetes"). The agent
runs four phases, with the last two gated on user confirmation:

**Phase 1 — Clarify.** Asks 2-4 sharp clarifying questions if the brief
is vague. (Role title, location, seniority, must-have skills,
employment type, timeline.)

**Phase 2 — Market Research & Report.** Runs 4-8 web searches via MiniMax
to gather current salary data, job board listings, talent-supply
indicators, and recent market news. Delivers a structured report:
- Market summary
- Talent availability rating (Easy / Moderate / Hard / Very Hard)
- Demand signals
- Salary range by seniority, with source URLs and dates
- 3-5 live sample job postings
- Recommendations
- Honest caveats

Ends with a single follow-up question asking whether the user wants
Phase 3 + 4.

**Phase 3 — Candidate Personality Profile.** Runs only when the user
confirms. Grounded in 2-4 more web searches for the push factors
specific to this role/location/year. Delivers:
- 2-3 sentence framing of who is most likely to be open to a move
- 5-7 personality traits / motivational drivers, each with the
  underlying motivation and what it means for how you should talk
  to them
- 3-4 anti-personas (people not worth recruiting)
- Likely objections a JD will need to address upfront
- A 1-sentence "candidate voice" used by Phase 4

Ends with a follow-up asking whether to proceed to Phase 4, or
whether the user wants to tweak the profile first (e.g. "focus more
on remote-first candidates", "drop the career-pivoter slice").

**Phase 4 — Targeted Job Descriptions.** Runs only when the user
approves the profile. Generates 3-5 JDs, each tuned to a different
slice of the personality profile. For each JD:
- A specific working title (not generic)
- A 1-sentence headline / hook
- Full job description in board-ready format (intro / what you'll do /
  what we're looking for / what we offer / how to apply)
- A 1-2 sentence rationale explaining which personality slice it
  targets and what trade-off the user is making

Ends with a Posting Strategy note (which boards for which slice, what
to A/B test, response rate expectations, legal/compliance to watch
— e.g. EU Pay Transparency Directive).

Phases 1+2 run automatically on the first turn. Phases 3+4 are
gated — the user can say "yes" / "go ahead" to proceed, or steer
("focus more on senior folks", "skip the mission-driven slice, we
already have a values pitch"). The profile can be iterated as many
times as the user wants before any JDs are written.

## Architecture

```
+-----------------+      SSE (text/event-stream)      +---------------+
|   Browser UI    |  <------------------------------>  |  FastAPI app  |
|  (vanilla JS)   |                                    |  (app.py)     |
+-----------------+                                    +-------+-------+
                                                                 |
                                       +-------------------------+----------------------+
                                       |                         |                      |
                                 +-----v-----+            +------v------+          +-----v-----+
                                 | sessions  |            |   agent     |          |  httpx    |
                                 |   .py     |            |    .py      |          |           |
                                 +-----------+            +------+------+          +-----+-----+
                                 data/sessions/                 |                     |
                                  *.json                 agent loop                  |
                                       |              (custom tool dispatch)          |
                                       v                                           |
                                +------+------+                                    |
                                |  MiniMax    |                                    |
                                |  endpoints  |<-----------------------------------+
                                +-------------+
                                  |          |
                       /v1/messages       /v1/coding_plan/search
                  (Anthropic compat,    (Bearer auth,
                   x-api-key, SSE)      JSON POST, JSON out)
```

- **Frontend:** Single `index.html` + `app.js` + `styles.css`. No build,
  no framework. Markdown rendered client-side from a small built-in
  parser (no CDN dependencies).
- **Backend:** FastAPI. One process, no SDK dependencies. Sessions
  persisted to JSON files. Streaming via Server-Sent Events.
- **Agent:** Talks to MiniMax via two httpx calls. Wraps a custom
  `web_search` function tool — when the model emits `tool_use` for it,
  we hit the search endpoint and feed results back as `tool_result`,
  letting the model iterate until it's satisfied and emits a final
  assistant message.

## Setup

```bash
cd job-research-agent
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Add your MiniMax API key (get one at https://platform.minimax.io)
echo 'MINIMAX_API_KEY=*** > .env

# Optional: override the model or endpoint
# echo 'JOB_MODEL=MiniMax-M3' >> .env
# echo 'MINIMAX_CHAT_BASE=https://api.minimax.io/anthropic' >> .env
# echo 'MINIMAX_SEARCH_BASE=https://api.minimax.io' >> .env

.venv/bin/uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>.

## Docker

```bash
docker build -t jobscope .
docker run -p 8000:8000 -e MINIMAX_API_KEY=*** \
  -v $PWD/data:/app/data jobscope
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MINIMAX_API_KEY` | _required_ | Your MiniMax API key. `ANTHROPIC_API_KEY` is also accepted as a fallback. |
| `JOB_MODEL` | `MiniMax-M3` | MiniMax model to use |
| `MINIMAX_CHAT_BASE` | `https://api.minimax.io/anthropic` | MiniMax's Anthropic-compatible chat endpoint |
| `MINIMAX_SEARCH_BASE` | `https://api.minimax.io` | MiniMax's search endpoint base URL |
| `JOBAGENT_DATA_DIR` | `data/sessions` | Where session JSON files live |
| `LOG_LEVEL` | `INFO` | Python log level |

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | Liveness + config check |
| `GET`  | `/api/sessions` | List sessions, newest first |
| `POST` | `/api/sessions` | Create a new session (optional `title`) |
| `GET`  | `/api/sessions/{id}` | Read a session (with full message history) |
| `PATCH` | `/api/sessions/{id}` | Rename a session (`{"title": "..."}`) |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `POST` | `/api/chat` | Send a turn; streams SSE events |

### SSE event types from `/api/chat`

- `search_start` — `{type, query}` — agent is about to run a web search
- `search_done` — `{type, query}` — search returned
- `text_delta` — `{type, text}` — model is typing; append to message
- `message_done` — `{type, text}` — final assistant message text
- `error` — `{type, message}` — agent failed
- `end` — terminal marker; the stream is closing

## Sample session

A real 3-turn session against the live MiniMax-M3 endpoint took
~4 minutes wall-clock and produced:

- **Phase 2 report:** 9,034 chars covering market summary, talent
  availability, demand signals, salary table by seniority (with 6
  cited sources), 5 live sample job postings, recommendations, caveats
- **Phase 3 profile:** 6,961 chars covering the open-to-move pool,
  4 personality traits (Stagnation-driven, Comp-blocked,
  Mission-driven, Oncall-burnt) with what each means for messaging,
  3 anti-personas, 4 likely objections, and a candidate voice
- **Phase 4 JDs:** 12,996 chars of board-ready JDs (3 of them, each
  ~500 words) plus a per-JD rationale and a Posting Strategy note
  with board recommendations, A/B test ideas, response rate
  expectations, and EU Pay Transparency Directive compliance notes

Total: 10 web searches in Phase 2, 6 in Phase 3, 0 in Phase 4.

## Testing

Run the end-to-end smoke test (no API key needed; mocks both endpoints
and scripts the multi-turn Phase 1→2→3→4 flow):

```bash
.venv/bin/python -m tests.smoke
```

## Limitations & next steps

- **Single-node only.** Sessions are JSON files with an in-process
  lock. If you run multiple uvicorn workers, switch to a real DB.
- **No auth.** Add a reverse proxy (oauth2-proxy, Cloudflare Access,
  etc.) if you expose this to the internet.
- **No streaming retry.** If the connection drops mid-turn, you have
  to re-send the message. Anthropic's prompt caching would help here
  if you start paying attention to token costs.
- **Reports aren't compared.** Each session is independent. Future
  work: let the user open two reports side-by-side or build a
  "compare two roles" view.
- **The `web_search` tool is a thin wrapper.** Results are returned
  as raw titles + snippets. If you want rich parsing, fetching page
  contents, citation cleanup, etc. — that would live in `agent.py`'s
  `web_search()` function.

## Project layout

```
job-research-agent/
├── backend/
│   ├── __init__.py
│   ├── agent.py        # the research loop + SSE parser + tool dispatch
│   ├── app.py          # FastAPI routes
│   └── sessions.py     # JSON file session storage
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js          # chat UI + markdown renderer
├── data/
│   └── sessions/       # one *.json per session
├── tests/
│   ├── __init__.py
│   └── smoke.py        # end-to-end test with mocked MiniMax endpoints
├── .env                # MINIMAX_API_KEY=*** (you create this)
├── requirements.txt
├── Dockerfile
└── README.md
```
