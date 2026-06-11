FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
RUN mkdir -p /app/data/sessions

ENV JOBAGENT_DATA_DIR=/app/data/sessions
ENV PYTHONUNBUFFERED=1
# Override these at runtime with -e:
#   MINIMAX_API_KEY, JOB_MODEL, MINIMAX_CHAT_BASE, MINIMAX_SEARCH_BASE

EXPOSE 8000

# Single uvicorn worker — sessions are file-locked per process. Scale by
# running multiple containers behind a reverse proxy that routes by
# session_id (sticky), or switch to a real DB first.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
