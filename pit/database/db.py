"""SQLite connection management.

All SQL lives in the repository modules of this package and uses ``?`` parameters and
standard SQL so a PostgreSQL backend can be added behind the same interface later.
Connections are short-lived (one per unit of work) because Streamlit and the scheduler use
threads and ``sqlite3`` connections must not be shared across threads.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

SCHEMA_VERSION = "1"


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._init_lock = threading.Lock()
        self._initialised = False

    # ------------------------------------------------------------------ connections
    def _connect(self) -> sqlite3.Connection:
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        if str(self.path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Autocommit connection (each statement is its own transaction)."""
        self.initialise()
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic unit of work: commits on success, rolls back on any exception."""
        self.initialise()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ schema
    def initialise(self) -> None:
        if self._initialised:
            return
        with self._init_lock:
            if self._initialised:
                return
            schema = resources.files("pit.database").joinpath("schema.sql").read_text("utf-8")
            conn = self._connect()
            try:
                conn.executescript(schema)
                conn.execute(
                    "INSERT INTO schema_info(key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (SCHEMA_VERSION,),
                )
            finally:
                conn.close()
            self._initialised = True

    # ------------------------------------------------------------------ helpers
    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else None

    def frame(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            return pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=cols)

    def integrity_check(self) -> dict[str, Any]:
        """Run SQLite's integrity check and report basic table counts."""
        with self.connect() as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            tables = [
                "candles",
                "forecasts",
                "forecast_evaluations",
                "feature_snapshots",
                "paper_transactions",
                "llm_calls",
            ]
            counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"integrity": result, "counts": counts, "size_bytes": size, "path": str(self.path)}
