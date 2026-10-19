"""Backtest runs and out-of-sample research declarations."""

from __future__ import annotations

import json
import uuid
from typing import Any

from pit.common.timeutils import iso, utcnow
from pit.database.db import Database


class ResearchRepository:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ declarations
    def declare(self, config_version: str, holdout_start: str, holdout_end: str, hypothesis: str) -> str:
        """Pre-register a configuration and an untouched holdout period for one OOS test."""
        declaration_id = "decl_" + uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO research_declarations(declaration_id, created_at, config_version, holdout_start, holdout_end, "
            "hypothesis) VALUES (?, ?, ?, ?, ?, ?)",
            (declaration_id, iso(utcnow()), config_version, holdout_start, holdout_end, hypothesis),
        )
        return declaration_id

    def declaration(self, declaration_id: str) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM research_declarations WHERE declaration_id = ?", (declaration_id,))

    def declarations(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM research_declarations ORDER BY created_at DESC")

    def consume_declaration(self, declaration_id: str, run_id: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute(
                "UPDATE research_declarations SET consumed_by_run = ? WHERE declaration_id = ? AND consumed_by_run IS NULL",
                (run_id, declaration_id),
            )
            return cur.rowcount == 1

    # ------------------------------------------------------------------ runs
    def create_run(self, name: str, validation_type: str, start_ts: str, end_ts: str, data_source: str,
                   data_mode: str, config_version: str, config: dict[str, Any], declaration_id: str | None = None,
                   notes: str | None = None) -> str:
        run_id = "bt_" + uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO backtest_runs(run_id, name, created_at, status, validation_type, declaration_id, start_ts, end_ts, "
            "data_source, data_mode, config_version, config_json, notes) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, name, iso(utcnow()), validation_type, declaration_id, start_ts, end_ts, data_source, data_mode,
             config_version, json.dumps(config, sort_keys=True), notes),
        )
        return run_id

    def finish_run(self, run_id: str, status: str, summary: dict[str, Any] | None = None) -> None:
        self.db.execute(
            "UPDATE backtest_runs SET status = ?, finished_at = ?, summary_json = ? WHERE run_id = ?",
            (status, iso(utcnow()), json.dumps(summary or {}, default=str), run_id),
        )

    def run(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,))
        if row and row.get("summary_json"):
            row["summary"] = json.loads(row["summary_json"])
        return row

    def runs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["summary"] = json.loads(r["summary_json"]) if r.get("summary_json") else {}
        return rows
