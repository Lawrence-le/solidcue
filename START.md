# 1. Frontend (studio/) — note it's pnpm, per pnpm-lock.yaml

pnpm dev # → Vite on :5173, proxies /api → :8000

# 2. LangGraph Server (runs/threads/streaming)

uv run langgraph dev --port 2024 --allow-blocking

# 3. Companion FastAPI (agent CRUD)

uv run uvicorn solidcue.api.main:app --port 8000
