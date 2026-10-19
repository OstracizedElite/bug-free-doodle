#!/usr/bin/env bash
# Run the PIT scheduler in the foreground (use systemd/supervisor for unattended operation).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m pit run "$@"
