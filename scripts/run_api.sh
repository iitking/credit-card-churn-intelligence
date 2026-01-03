#!/usr/bin/env bash
# Launch FastAPI REST API server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Launching FastAPI server at http://localhost:8000 (Docs: http://localhost:8000/docs) ..."
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
