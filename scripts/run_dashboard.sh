#!/usr/bin/env bash
# Start the Streamlit dashboard on localhost (put an authenticating proxy in front for remote access).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/streamlit run pit/dashboard/app.py --server.address "${PIT_DASHBOARD_ADDRESS:-127.0.0.1}" \
  --server.port "${PIT_DASHBOARD_PORT:-8501}" "$@"
