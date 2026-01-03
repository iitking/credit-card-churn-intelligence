#!/usr/bin/env bash
# Launch Streamlit web dashboard
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Launching Streamlit Dashboard at http://localhost:8501 ..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
