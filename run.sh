#!/usr/bin/env bash
# Start the Z-Image-Turbo web UI (FastAPI backend + Vite frontend).
#
#   ./run.sh          start both (backend :8000, frontend :5173)
#   ./run.sh backend  backend only
#   ./run.sh frontend frontend only
#
# The backend loads the INT4 model at startup (~25 s) and keeps it resident.
set -euo pipefail

ROOT="/home/peanut/Projects/strata-diffusion"
VENV="$ROOT/.venv"
MODE="${1:-all}"

start_backend() {
  cd "$ROOT/backend"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  echo "backend  -> http://127.0.0.1:8000  (loading model, ~25s)"
  exec python -m uvicorn main:app --host 127.0.0.1 --port 8000
}

start_frontend() {
  cd "$ROOT/frontend"
  echo "frontend -> http://127.0.0.1:5173"
  exec npm run dev
}

case "$MODE" in
  backend)  start_backend ;;
  frontend) start_frontend ;;
  all)
    start_backend &
    BACK=$!
    # Wait for the model to be up before the frontend proxies to it.
    for _ in $(seq 1 90); do
      sleep 2
      if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then break; fi
    done
    start_frontend
    ;;
  *) echo "usage: $0 [all|backend|frontend]" >&2; exit 1 ;;
esac
