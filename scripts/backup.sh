#!/usr/bin/env bash
# Daily backup helper (cron): keeps the 14 most recent backups.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m pit backup
ls -1t var/backups/pit-*.db 2>/dev/null | tail -n +15 | xargs -r rm --
