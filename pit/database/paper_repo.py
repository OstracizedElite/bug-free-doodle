"""Paper-trading persistence. Kept entirely separate from forecast records."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd

from pit.common.timeutils import iso, utcnow
from pit.database.db import Database
from pit.paper_trading.engine import AccountState, PaperEvents


class PaperRepository:
    def __init__(self, db: Database):
        self.db = db

    def create_account(self, name: str, origin: str, run_id: str, config_version: str, config: dict[str, Any],
                       base_currency: str, initial_balance: float) -> str:
        account_id = "acct_" + uuid.uuid4().hex[:16]
        state = AccountState.initial(initial_balance)
        self.db.execute(
            "INSERT INTO paper_accounts(account_id, name, origin, run_id, config_version, config_json, base_currency, "
            "initial_balance, created_at, last_processed_min, state_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (account_id, name, origin, run_id, config_version, json.dumps(config, sort_keys=True), base_currency,
             initial_balance, iso(utcnow()), None, state.to_json()),
        )
        return account_id

    def find_account(self, origin: str, config_version: str, run_id: str = "") -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM paper_accounts WHERE origin = ? AND config_version = ? AND run_id = ? "
            "ORDER BY created_at LIMIT 1",
            (origin, config_version, run_id),
        )

    def get_account(self, account_id: str) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM paper_accounts WHERE account_id = ?", (account_id,))

    def accounts(self, origin: str | None = None) -> list[dict[str, Any]]:
        if origin:
            return self.db.query("SELECT * FROM paper_accounts WHERE origin = ? ORDER BY created_at DESC", (origin,))
        return self.db.query("SELECT * FROM paper_accounts ORDER BY created_at DESC")

    def load_state(self, account_id: str) -> AccountState:
        row = self.get_account(account_id)
        if row is None:
            raise KeyError(account_id)
        return AccountState.from_json(row["state_json"])

    def persist(self, account_id: str, state: AccountState, events: PaperEvents) -> None:
        """Atomically store a processing step: new/closed positions, transactions, equity, state."""
        with self.db.transaction() as conn:
            for p in events.opened:
                cols = list(p)
                conn.execute(
                    f"INSERT INTO paper_positions({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                    tuple(p[c] for c in cols),
                )
            for c in events.closed:
                fields = [k for k in c if k != "position_id"]
                conn.execute(
                    f"UPDATE paper_positions SET {', '.join(f'{k} = ?' for k in fields)} WHERE position_id = ?",
                    tuple(c[k] for k in fields) + (c["position_id"],),
                )
            if events.transactions:
                cols = list(events.transactions[0])
                conn.executemany(
                    f"INSERT INTO paper_transactions({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                    [tuple(t[c] for c in cols) for t in events.transactions],
                )
            if events.equity:
                cols = list(events.equity[0])
                conn.executemany(
                    f"INSERT OR REPLACE INTO paper_equity({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                    [tuple(e[c] for c in cols) for e in events.equity],
                )
            conn.execute(
                "UPDATE paper_accounts SET state_json = ?, last_processed_min = ? WHERE account_id = ?",
                (state.to_json(), state.last_min, account_id),
            )

    def positions(self, account_id: str, status: str | None = None) -> pd.DataFrame:
        if status:
            return self.db.frame(
                "SELECT * FROM paper_positions WHERE account_id = ? AND status = ? ORDER BY entry_ts", (account_id, status)
            )
        return self.db.frame("SELECT * FROM paper_positions WHERE account_id = ? ORDER BY entry_ts", (account_id,))

    def transactions(self, account_id: str, limit: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM paper_transactions WHERE account_id = ? ORDER BY seq DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.db.frame(sql, (account_id,))

    def equity(self, account_id: str) -> pd.DataFrame:
        return self.db.frame("SELECT * FROM paper_equity WHERE account_id = ? ORDER BY ts_min", (account_id,))
