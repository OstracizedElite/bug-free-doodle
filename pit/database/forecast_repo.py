"""Forecast, target, snapshot and evaluation persistence.

Forecast rows are inserted exactly once and never modified (database triggers reject
UPDATE/DELETE). Evaluations are appended to separate tables.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

import pandas as pd

from pit.common.timeutils import iso, utcnow
from pit.database.db import Database
from pit.forecasting.models import Forecast, ForecastResult, ForecastTarget

_FORECAST_COLS = list(Forecast.model_fields)
_TARGET_COLS = list(ForecastTarget.model_fields)


def _target_row(t: ForecastTarget) -> tuple:
    d = t.model_dump()
    d["is_primary"] = int(d["is_primary"])
    d["is_mirror"] = int(d["is_mirror"])
    return tuple(d[c] for c in _TARGET_COLS)


class ForecastRepository:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ writes
    def _insert(self, conn: sqlite3.Connection, result: ForecastResult, store_snapshot: bool, ignore: bool) -> bool:
        f = result.forecast if result.forecast.content_hash else result.forecast.with_hash()
        verb = "INSERT OR IGNORE" if ignore else "INSERT"
        if store_snapshot and result.snapshot is not None:
            snap = result.snapshot
            conn.execute(
                "INSERT OR IGNORE INTO feature_snapshots(snapshot_id, instrument, as_of_ts, input_hash, features_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (snap.snapshot_id, snap.instrument, f.as_of_ts, snap.input_hash,
                 json.dumps(snap.payload(), sort_keys=True), f.created_at),
            )
        cur = conn.execute(
            f"{verb} INTO forecasts({', '.join(_FORECAST_COLS)}) VALUES ({', '.join('?' for _ in _FORECAST_COLS)})",
            tuple(getattr(f, c) for c in _FORECAST_COLS),
        )
        if cur.rowcount == 0:
            return False
        if result.targets:
            conn.executemany(
                f"INSERT INTO forecast_targets({', '.join(_TARGET_COLS)}) VALUES ({', '.join('?' for _ in _TARGET_COLS)})",
                [_target_row(t) for t in result.targets],
            )
        return True

    def insert(self, result: ForecastResult, store_snapshot: bool = True) -> bool:
        """Insert one forecast atomically. Returns False if its slot was already forecast."""
        try:
            with self.db.transaction() as conn:
                return self._insert(conn, result, store_snapshot, ignore=False)
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc):
                return False
            raise

    def insert_many(self, results: Iterable[ForecastResult], store_snapshot: bool = False) -> int:
        n = 0
        with self.db.transaction() as conn:
            for r in results:
                n += int(self._insert(conn, r, store_snapshot, ignore=True))
        return n

    # ------------------------------------------------------------------ reads
    def get(self, forecast_id: str) -> Forecast | None:
        row = self.db.query_one(f"SELECT {', '.join(_FORECAST_COLS)} FROM forecasts WHERE forecast_id = ?", (forecast_id,))
        return Forecast(**row) if row else None

    def targets(self, forecast_id: str) -> list[ForecastTarget]:
        rows = self.db.query(f"SELECT {', '.join(_TARGET_COLS)} FROM forecast_targets WHERE forecast_id = ?", (forecast_id,))
        return [ForecastTarget(**r) for r in rows]

    def targets_for(self, forecast_ids: list[str]) -> dict[str, list[ForecastTarget]]:
        out: dict[str, list[ForecastTarget]] = {fid: [] for fid in forecast_ids}
        for chunk_start in range(0, len(forecast_ids), 500):
            chunk = forecast_ids[chunk_start:chunk_start + 500]
            rows = self.db.query(
                f"SELECT {', '.join(_TARGET_COLS)} FROM forecast_targets WHERE forecast_id IN ({', '.join('?' for _ in chunk)})",
                chunk,
            )
            for r in rows:
                out[r["forecast_id"]].append(ForecastTarget(**r))
        return out

    def snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM feature_snapshots WHERE snapshot_id = ?", (snapshot_id,))
        if row:
            row["features"] = json.loads(row.pop("features_json"))
        return row

    def slot_exists(self, origin: str, run_id: str, instrument: str, slot_min: int, model_type: str,
                    config_version: str) -> bool:
        return bool(
            self.db.scalar(
                "SELECT 1 FROM forecasts WHERE origin = ? AND run_id = ? AND instrument = ? AND slot_min = ? "
                "AND model_type = ? AND config_version = ? LIMIT 1",
                (origin, run_id, instrument, slot_min, model_type, config_version),
            )
        )

    def latest(self, origin: str = "live", model_type: str = "deterministic", run_id: str = "") -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM v_forecast_status WHERE origin = ? AND model_type = ? AND run_id = ? "
            "ORDER BY slot_min DESC, horizon_minutes LIMIT 1",
            (origin, model_type, run_id),
        )

    def last_slot_min(self, origin: str = "live", instrument: str | None = None) -> int | None:
        if instrument:
            v = self.db.scalar("SELECT MAX(slot_min) FROM forecasts WHERE origin = ? AND instrument = ?", (origin, instrument))
        else:
            v = self.db.scalar("SELECT MAX(slot_min) FROM forecasts WHERE origin = ?", (origin,))
        return int(v) if v is not None else None

    def frame(
        self,
        origin: str | None = "live",
        run_id: str | None = None,
        model_type: str | None = None,
        start_min: int | None = None,
        end_min: int | None = None,
        limit: int | None = None,
        data_mode: str | None = None,
    ) -> pd.DataFrame:
        where, params = [], []
        for col, val in (("origin", origin), ("run_id", run_id), ("model_type", model_type), ("data_mode", data_mode)):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        if start_min is not None:
            where.append("slot_min >= ?")
            params.append(start_min)
        if end_min is not None:
            where.append("slot_min < ?")
            params.append(end_min)
        sql = "SELECT * FROM v_forecast_status"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY slot_min DESC, horizon_minutes, model_type"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.db.frame(sql, params)

    def pending_evaluation(
        self, evaluator_version: str, max_expiry_min: int, origin: str | None = None, run_id: str | None = None,
        limit: int = 5000,
    ) -> list[Forecast]:
        where = ["f.expiry_min <= ?", "e.forecast_id IS NULL"]
        params: list[Any] = [evaluator_version, max_expiry_min]
        if origin is not None:
            where.append("f.origin = ?")
            params.append(origin)
        if run_id is not None:
            where.append("f.run_id = ?")
            params.append(run_id)
        cols = ", ".join(f"f.{c}" for c in _FORECAST_COLS)
        rows = self.db.query(
            f"SELECT {cols} FROM forecasts f LEFT JOIN forecast_evaluations e "
            f"ON e.forecast_id = f.forecast_id AND e.evaluator_version = ? "
            f"WHERE {' AND '.join(where)} ORDER BY f.expiry_min LIMIT {int(limit)}",
            params,
        )
        return [Forecast(**r) for r in rows]

    # ------------------------------------------------------------------ evaluations
    def insert_evaluations(self, evaluations: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> int:
        if not evaluations:
            return 0
        cols = list(evaluations[0])
        tcols = ["forecast_id", "evaluator_version", "label", "hit", "first_touch_ts", "minutes_to_touch"]
        with self.db.transaction() as conn:
            before = conn.total_changes
            conn.executemany(
                f"INSERT OR IGNORE INTO forecast_evaluations({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [tuple(e[c] for c in cols) for e in evaluations],
            )
            n = conn.total_changes - before
            if target_rows:
                conn.executemany(
                    f"INSERT OR IGNORE INTO target_evaluations({', '.join(tcols)}) VALUES ({', '.join('?' for _ in tcols)})",
                    [tuple(t[c] for c in tcols) for t in target_rows],
                )
        return n

    def target_evaluations(self, origin: str | None = None, run_id: str | None = None) -> pd.DataFrame:
        where, params = [], []
        if origin is not None:
            where.append("f.origin = ?")
            params.append(origin)
        if run_id is not None:
            where.append("f.run_id = ?")
            params.append(run_id)
        sql = (
            "SELECT t.*, ft.atr_multiple, ft.distance, ft.direction AS target_direction, ft.is_mirror, "
            "f.direction, f.candidate_direction, f.model_type, f.horizon_minutes, f.slot_min "
            "FROM target_evaluations t JOIN forecasts f ON f.forecast_id = t.forecast_id "
            "JOIN forecast_targets ft ON ft.forecast_id = t.forecast_id AND ft.label = t.label"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self.db.frame(sql, params)

    # ------------------------------------------------------------------ integrity
    def verify_hashes(self, limit: int | None = None) -> dict[str, Any]:
        sql = f"SELECT {', '.join(_FORECAST_COLS)} FROM forecasts ORDER BY slot_min DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        bad = []
        rows = self.db.query(sql)
        for r in rows:
            f = Forecast(**r)
            if f.computed_hash() != f.content_hash:
                bad.append(f.forecast_id)
        return {"checked": len(rows), "mismatched": bad, "checked_at": iso(utcnow())}
