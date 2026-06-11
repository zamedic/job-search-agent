# JobScope — Job Market Research Agent

A small web app that turns a free-text hiring question into a structured
job-market report. Powered by **MiniMax-M3** (via the Anthropic-compatible
chat endpoint) plus a custom `web_search` function tool that hits
MiniMax's `coding_plan/search` endpoint.

The user describes a role they're trying to fill (e.g. "I need a senior
backend engineer in Berlin, hybrid, Go and Kubernetes"). The agent:

1. Asks 2-4 sharp clarifying questions if the brief is vague.
2. Runs 4-8 web searches via MiniMax to gather current salary data,
   job board listings, talent-supply indicators, and recent market news.
3. Delivers a structured report with:
   - Market summary
   - Talent availability rating (Easy / Moderate / Hard / Very Hard)
   - Demand signals
   - Salary range by seniority, with source URLs and dates
   - 3-5 live sample job postings
   - Recommendations
   - Honest caveats

The user can ask follow-ups and the agent will do more targeted research.

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

## Testing

Run the end-to-end smoke test (no API key needed; mocks both endpoints):

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
