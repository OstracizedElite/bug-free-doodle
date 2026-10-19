"""Immutable LLM call log (inputs, responses, tokens, cost, latency, errors)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from pit.database.db import Database

_COLS = [
    "call_id", "forecast_id", "created_at", "provider", "model", "prompt_version", "cache_key", "cached",
    "input_json", "response_json", "status", "error", "input_tokens", "output_tokens", "cost_usd", "latency_ms",
]


class LLMRepository:
    def __init__(self, db: Database):
        self.db = db

    def insert(self, row: dict[str, Any]) -> None:
        self.db.execute(
            f"INSERT INTO llm_calls({', '.join(_COLS)}) VALUES ({', '.join('?' for _ in _COLS)})",
            tuple(row.get(c) for c in _COLS),
        )

    def cached_response(self, cache_key: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT response_json FROM llm_calls WHERE cache_key = ? AND status = 'ok' ORDER BY created_at LIMIT 1",
            (cache_key,),
        )
        return json.loads(row["response_json"]) if row and row["response_json"] else None

    def spend_since(self, since_iso: str) -> float:
        return float(self.db.scalar("SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE created_at >= ?",
                                    (since_iso,)) or 0.0)

    def usage(self, since_iso: str) -> pd.DataFrame:
        return self.db.frame(
            "SELECT provider, model, status, COUNT(*) AS calls, SUM(cached) AS cached, "
            "SUM(COALESCE(input_tokens, 0)) AS input_tokens, SUM(COALESCE(output_tokens, 0)) AS output_tokens, "
            "SUM(cost_usd) AS cost_usd, AVG(latency_ms) AS mean_latency_ms "
            "FROM llm_calls WHERE created_at >= ? GROUP BY provider, model, status",
            (since_iso,),
        )

    def recent(self, limit: int = 50) -> pd.DataFrame:
        return self.db.frame(
            "SELECT call_id, forecast_id, created_at, provider, model, prompt_version, cached, status, error, "
            "input_tokens, output_tokens, cost_usd, latency_ms FROM llm_calls ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
