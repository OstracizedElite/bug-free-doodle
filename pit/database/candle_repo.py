"""Candle, quote and gap persistence."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from pit.common.timeutils import iso, iso_min, utcnow
from pit.data.models import CANDLE_COLUMNS, Quote, conform_candle_frame
from pit.data.quality import Gap
from pit.database.db import Database

_INSERT_COLS = ["instrument", "source", "ts", "ingested_at"] + CANDLE_COLUMNS


def _none_if_nan(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


class CandleRepository:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ candles
    def insert_candles(self, instrument: str, source: str, df: pd.DataFrame) -> int:
        """Insert validated candles. Existing rows are never overwritten (first write wins).

        Returns the number of rows actually inserted.
        """
        if df.empty:
            return 0
        frame = conform_candle_frame(df)
        now = iso(utcnow())
        minutes = [int(m) for m in frame["ts_min"].to_numpy()]
        others = frame[CANDLE_COLUMNS[1:]]
        others = others.astype(object).where(others.notna(), None)
        rows = [
            (instrument, source, iso_min(m), now, m, *(_none_if_nan(v) for v in vals))
            for m, vals in zip(minutes, others.itertuples(index=False, name=None))
        ]
        cols = ", ".join(_INSERT_COLS)
        marks = ", ".join("?" for _ in _INSERT_COLS)
        with self.db.transaction() as conn:
            before = conn.total_changes
            conn.executemany(f"INSERT OR IGNORE INTO candles({cols}) VALUES ({marks})", rows)
            return conn.total_changes - before

    def load(
        self,
        instrument: str,
        source: str,
        start_min: int | None = None,
        end_min: int | None = None,
        limit_last: int | None = None,
    ) -> pd.DataFrame:
        """Candles with ``start_min <= ts_min < end_min`` ordered by time."""
        where = ["instrument = ?", "source = ?"]
        params: list[Any] = [instrument, source]
        if start_min is not None:
            where.append("ts_min >= ?")
            params.append(int(start_min))
        if end_min is not None:
            where.append("ts_min < ?")
            params.append(int(end_min))
        cols = ", ".join(CANDLE_COLUMNS)
        sql = f"SELECT {cols} FROM candles WHERE {' AND '.join(where)}"
        if limit_last:
            sql = f"SELECT * FROM ({sql} ORDER BY ts_min DESC LIMIT {int(limit_last)}) ORDER BY ts_min"
        else:
            sql += " ORDER BY ts_min"
        df = self.db.frame(sql, params)
        if df.empty:
            return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
        return conform_candle_frame(df)

    def latest_ts_min(self, instrument: str, source: str) -> int | None:
        v = self.db.scalar("SELECT MAX(ts_min) FROM candles WHERE instrument = ? AND source = ?", (instrument, source))
        return int(v) if v is not None else None

    def earliest_ts_min(self, instrument: str, source: str) -> int | None:
        v = self.db.scalar("SELECT MIN(ts_min) FROM candles WHERE instrument = ? AND source = ?", (instrument, source))
        return int(v) if v is not None else None

    def last_close_before(self, instrument: str, source: str, ts_min: int) -> float | None:
        v = self.db.scalar(
            "SELECT close FROM candles WHERE instrument = ? AND source = ? AND ts_min < ? ORDER BY ts_min DESC LIMIT 1",
            (instrument, source, int(ts_min)),
        )
        return float(v) if v is not None else None

    def existing_minutes(self, instrument: str, source: str, start_min: int, end_min: int) -> np.ndarray:
        rows = self.db.query(
            "SELECT ts_min FROM candles WHERE instrument = ? AND source = ? AND ts_min >= ? AND ts_min < ?",
            (instrument, source, int(start_min), int(end_min)),
        )
        return np.array([r["ts_min"] for r in rows], dtype="int64")

    def sources(self) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT instrument, source, data_mode, COUNT(*) AS candles, MIN(ts) AS first_ts, MAX(ts) AS last_ts, "
            "SUM(CASE WHEN quality = 'suspect' THEN 1 ELSE 0 END) AS suspect, "
            "SUM(CASE WHEN spread_source = 'assumed' THEN 1 ELSE 0 END) AS assumed_spread "
            "FROM candles GROUP BY instrument, source, data_mode ORDER BY instrument, source"
        )

    def suspect_since(self, instrument: str, source: str, since_min: int, until_min: int) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM candles WHERE instrument = ? AND source = ? AND ts_min >= ? AND ts_min < ? "
                "AND quality = 'suspect'",
                (instrument, source, int(since_min), int(until_min)),
            )
            or 0
        )

    # ------------------------------------------------------------------ quotes
    def insert_quotes(self, quotes: Iterable[Quote]) -> int:
        now = iso(utcnow())
        rows = [
            (
                q.instrument,
                q.source,
                int(q.ts.timestamp() * 1000),
                iso(q.ts),
                q.bid,
                q.ask,
                q.mid,
                q.spread,
                q.data_mode.value if hasattr(q.data_mode, "value") else str(q.data_mode),
                now,
            )
            for q in quotes
        ]
        if not rows:
            return 0
        with self.db.transaction() as conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO quotes(instrument, source, ts_ms, ts, bid, ask, mid, spread, data_mode, received_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return conn.total_changes - before

    def load_quotes(self, instrument: str, source: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        return self.db.frame(
            "SELECT ts_ms, bid, ask, mid, spread FROM quotes WHERE instrument = ? AND source = ? "
            "AND ts_ms >= ? AND ts_ms < ? ORDER BY ts_ms",
            (instrument, source, int(start_ms), int(end_ms)),
        )

    # ------------------------------------------------------------------ gaps
    def record_gaps(self, instrument: str, source: str, gaps: Iterable[Gap]) -> int:
        now = iso(utcnow())
        rows = [
            (instrument, source, g.start_min, iso_min(g.start_min), iso_min(g.end_min), g.missing_minutes, now)
            for g in gaps
        ]
        if not rows:
            return 0
        with self.db.transaction() as conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO data_gaps(instrument, source, gap_start_min, gap_start, gap_end, "
                "missing_minutes, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return conn.total_changes - before

    def gaps(self, instrument: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        if instrument:
            return self.db.query(
                "SELECT * FROM data_gaps WHERE instrument = ? ORDER BY gap_start_min DESC LIMIT ?", (instrument, limit)
            )
        return self.db.query("SELECT * FROM data_gaps ORDER BY gap_start_min DESC LIMIT ?", (limit,))
