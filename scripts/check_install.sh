#!/usr/bin/env bash
# Verify an installation end to end on synthetic data, in a throwaway database.
set -euo pipefail
cd "$(dirname "$0")/.."
DB="var/check-install.db"
rm -f "$DB" "$DB-wal" "$DB-shm"
.venv/bin/python -m pytest -q
.venv/bin/python -m pit --db "$DB" --log-level WARNING demo --days 7
.venv/bin/python -m pit --db "$DB" --log-level WARNING verify
echo "OK: tests passed, synthetic demo ran and integrity verified ($DB)"
