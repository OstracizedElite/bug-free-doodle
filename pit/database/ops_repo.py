"""Operational records: config versions, feed status, errors, usage, heartbeats, scheduler runs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pit.common.timeutils import iso, utcnow
from pit.database.db import Database


class OpsRepository:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ config versions
    def register_config(self, version: str, scope: str, config: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO config_versions(config_version, scope, config_json, first_seen_at) "
            "VALUES (?, ?, ?, ?)",
            (version, scope, json.dumps(config, sort_keys=True), iso(utcnow())),
        )

    def get_config(self, version: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM config_versions WHERE config_version = ?", (version,))
        if row:
            row["config"] = json.loads(row.pop("config_json"))
        return row

    # ------------------------------------------------------------------ feed status
    def upsert_feed_status(
        self,
        instrument: str,
        source: str,
        status: str,
        data_mode: str | None = None,
        last_candle_ts: str | None = None,
        success: bool = False,
        error: str | None = None,
        latency_seconds: float | None = None,
        now: datetime | None = None,
    ) -> None:
        now_s = iso(now or utcnow())
        existing = self.db.query_one(
            "SELECT * FROM feed_status WHERE instrument = ? AND source = ?", (instrument, source)
        ) or {}
        row = {
            "instrument": instrument,
            "source": source,
            "status": status,
            "data_mode": data_mode or existing.get("data_mode"),
            "last_candle_ts": last_candle_ts or existing.get("last_candle_ts"),
            "last_success_at": now_s if success else existing.get("last_success_at"),
            "last_error_at": now_s if error else existing.get("last_error_at"),
            "last_error": error if error else existing.get("last_error"),
            "latency_seconds": latency_seconds if latency_seconds is not None else existing.get("latency_seconds"),
            "updated_at": now_s,
        }
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        updates = ", ".join(f"{k} = excluded.{k}" for k in row if k not in ("instrument", "source"))
        self.db.execute(
            f"INSERT INTO feed_status({cols}) VALUES ({marks}) "
            f"ON CONFLICT(instrument, source) DO UPDATE SET {updates}",
            tuple(row.values()),
        )

    def feed_status(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM feed_status ORDER BY instrument, source")

    # ------------------------------------------------------------------ errors and usage
    def log_api_error(
        self,
        component: str,
        message: str,
        provider: str | None = None,
        endpoint: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO api_errors(component, provider, endpoint, status_code, message, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (component, provider, endpoint, status_code, message[:2000], iso(utcnow())),
        )

    def recent_api_errors(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM api_errors ORDER BY id DESC LIMIT ?", (limit,))

    def log_api_usage(
        self, component: str, provider: str, endpoint: str | None = None, units: float = 1, cost_usd: float = 0.0
    ) -> None:
        self.db.execute(
            "INSERT INTO api_usage(component, provider, endpoint, units, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (component, provider, endpoint, units, cost_usd, iso(utcnow())),
        )

    def usage_since(self, since_iso: str) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT component, provider, SUM(units) AS units, SUM(cost_usd) AS cost_usd, COUNT(*) AS calls "
            "FROM api_usage WHERE created_at >= ? GROUP BY component, provider",
            (since_iso,),
        )

    # ------------------------------------------------------------------ data quality
    def log_quality_event(
        self, instrument: str, source: str, event_type: str, severity: str, details: str, ts: str | None = None
    ) -> None:
        self.db.execute(
            "INSERT INTO data_quality_events(instrument, source, ts, event_type, severity, details, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (instrument, source, ts, event_type, severity, details[:2000], iso(utcnow())),
        )

    def recent_quality_events(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM data_quality_events ORDER BY id DESC LIMIT ?", (limit,))

    # ------------------------------------------------------------------ heartbeats / scheduler
    def heartbeat(self, component: str, details: dict[str, Any] | None = None, now: datetime | None = None) -> None:
        self.db.execute(
            "INSERT INTO heartbeats(component, last_beat, details) VALUES (?, ?, ?) "
            "ON CONFLICT(component) DO UPDATE SET last_beat = excluded.last_beat, details = excluded.details",
            (component, iso(now or utcnow()), json.dumps(details or {})),
        )

    def heartbeats(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM heartbeats ORDER BY component")

    def start_job(self, job: str, now: datetime | None = None) -> int:
        with self.db.connect() as conn:
            cur = conn.execute(
                "INSERT INTO scheduler_runs(job, started_at, status) VALUES (?, ?, 'running')",
                (job, iso(now or utcnow())),
            )
            return int(cur.lastrowid)

    def finish_job(self, run_id: int, status: str, message: str | None = None, now: datetime | None = None) -> None:
        self.db.execute(
            "UPDATE scheduler_runs SET finished_at = ?, status = ?, message = ? WHERE id = ?",
            (iso(now or utcnow()), status, (message or "")[:2000], run_id),
        )

    def recent_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM scheduler_runs ORDER BY id DESC LIMIT ?", (limit,))

    def last_job(self, job: str, status: str | None = None) -> dict[str, Any] | None:
        if status:
            return self.db.query_one(
                "SELECT * FROM scheduler_runs WHERE job = ? AND status = ? ORDER BY id DESC LIMIT 1", (job, status)
            )
        return self.db.query_one("SELECT * FROM scheduler_runs WHERE job = ? ORDER BY id DESC LIMIT 1", (job,))

    def record_missed_slot(self, instrument: str, slot_min: int, slot_ts: str, reason: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO missed_slots(instrument, slot_min, slot_ts, detected_at, reason) "
                "VALUES (?, ?, ?, ?, ?)",
                (instrument, slot_min, slot_ts, iso(utcnow()), reason),
            )
            return cur.rowcount > 0

    def missed_slots(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM missed_slots ORDER BY slot_min DESC LIMIT ?", (limit,))
