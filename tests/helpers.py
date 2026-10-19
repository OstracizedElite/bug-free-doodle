"""Deterministic test data builders with known expected results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pit.common.timeutils import to_epoch_min
from pit.data.models import conform_candle_frame

# Monday 2026-01-05 10:00 UTC: market open (05:00 New York, winter time).
MONDAY_10_UTC = to_epoch_min("2026-01-05T10:00:00Z")


def make_candles(
    closes,
    start_min: int = MONDAY_10_UTC,
    spread: float = 0.2,
    wick: float = 0.1,
    opens=None,
    highs=None,
    lows=None,
    step: int = 1,
) -> pd.DataFrame:
    """Build a 1-minute candle frame from closes. Open = previous close unless given."""
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    if opens is None:
        opens = np.concatenate([[closes[0]], closes[:-1]])
    opens = np.asarray(opens, dtype=float)
    if highs is None:
        highs = np.maximum(opens, closes) + wick
    if lows is None:
        lows = np.minimum(opens, closes) - wick
    highs, lows = np.asarray(highs, dtype=float), np.asarray(lows, dtype=float)
    half = spread / 2.0
    df = pd.DataFrame(
        {
            "ts_min": start_min + np.arange(n, dtype="int64") * step,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "bid_open": opens - half,
            "bid_high": highs - half,
            "bid_low": lows - half,
            "bid_close": closes - half,
            "ask_open": opens + half,
            "ask_high": highs + half,
            "ask_low": lows + half,
            "ask_close": closes + half,
            "spread": spread,
            "spread_source": "observed",
            "volume": 10.0,
            "quote_convention": "test_mid",
            "data_mode": "historical",
            "quality": "ok",
            "quality_notes": None,
        }
    )
    return conform_candle_frame(df)


def random_walk_closes(n: int, start: float = 2000.0, sigma: float = 0.5, drift: float = 0.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    return start + np.cumsum(drift + sigma * rng.standard_normal(n))
