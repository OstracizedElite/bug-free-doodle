"""Multi-timeframe bar series built from 1-minute candles.

Look-ahead safety: a bar covering ``[start, end)`` is *completed* only when
``end <= as_of``. Resampling is done once over all available data (a completed bar's OHLC
depends only on its own minutes, all of which precede ``as_of``) and :meth:`select`
returns only bars with ``end <= as_of``. The partially formed current bar is never used.

Reproducibility: feature calculations operate on a fixed number of the most recent
completed bars per timeframe (``features.lookback_bars``), so a forecast is identical
whether computed live from a short history or during replay over a long one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pit.config.settings import TIMEFRAME_MINUTES


@dataclass(frozen=True)
class BarSeries:
    tf: str
    minutes: int
    start: np.ndarray  # int64 epoch minutes
    end: np.ndarray  # int64 epoch minutes (exclusive)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    bid_close: np.ndarray
    ask_close: np.ndarray
    spread: np.ndarray
    count: np.ndarray  # number of 1m candles in the bar
    suspect: np.ndarray  # bool: bar contains a suspect 1m candle

    def __len__(self) -> int:
        return int(self.start.size)

    def slice(self, lo: int, hi: int) -> "BarSeries":
        return BarSeries(
            self.tf,
            self.minutes,
            *(arr[lo:hi] for arr in (
                self.start, self.end, self.open, self.high, self.low, self.close,
                self.bid_close, self.ask_close, self.spread, self.count, self.suspect,
            )),
        )

    def completed_index(self, as_of_min: int) -> int:
        """Number of bars with ``end <= as_of_min`` (bars are sorted by start)."""
        return int(np.searchsorted(self.end, as_of_min, side="right"))

    def select(self, as_of_min: int, lookback: int) -> "BarSeries":
        hi = self.completed_index(as_of_min)
        return self.slice(max(0, hi - lookback), hi)


def _bars_from_1m(df: pd.DataFrame) -> BarSeries:
    start = df["ts_min"].to_numpy(dtype="int64")
    return BarSeries(
        tf="1m",
        minutes=1,
        start=start,
        end=start + 1,
        open=df["open"].to_numpy(dtype="float64"),
        high=df["high"].to_numpy(dtype="float64"),
        low=df["low"].to_numpy(dtype="float64"),
        close=df["close"].to_numpy(dtype="float64"),
        bid_close=df["bid_close"].to_numpy(dtype="float64"),
        ask_close=df["ask_close"].to_numpy(dtype="float64"),
        spread=df["spread"].to_numpy(dtype="float64"),
        count=np.ones(start.size, dtype="int64"),
        suspect=(df["quality"].to_numpy() == "suspect") if "quality" in df else np.zeros(start.size, dtype=bool),
    )


def resample(base: BarSeries, tf: str) -> BarSeries:
    """Aggregate 1-minute bars into ``tf`` bars aligned to the UTC epoch (and therefore to
    UTC midnight for 4h and 12h)."""
    m = TIMEFRAME_MINUTES[tf]
    if m == 1:
        return base
    if len(base) == 0:
        empty_i = np.zeros(0, dtype="int64")
        empty_f = np.zeros(0, dtype="float64")
        return BarSeries(tf, m, empty_i, empty_i, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, empty_f,
                         empty_i, np.zeros(0, dtype=bool))
    keys = base.start - np.mod(base.start, m)
    starts_idx = np.flatnonzero(np.concatenate([[True], keys[1:] != keys[:-1]]))
    ends_idx = np.concatenate([starts_idx[1:], [keys.size]]) - 1
    bar_start = keys[starts_idx]
    return BarSeries(
        tf=tf,
        minutes=m,
        start=bar_start,
        end=bar_start + m,
        open=base.open[starts_idx],
        high=np.maximum.reduceat(base.high, starts_idx),
        low=np.minimum.reduceat(base.low, starts_idx),
        close=base.close[ends_idx],
        bid_close=base.bid_close[ends_idx],
        ask_close=base.ask_close[ends_idx],
        spread=base.spread[ends_idx],
        count=(ends_idx - starts_idx + 1).astype("int64"),
        suspect=np.logical_or.reduceat(base.suspect, starts_idx),
    )


class MarketFrames:
    """All configured timeframes for one instrument, resampled once."""

    def __init__(self, candles_1m: pd.DataFrame, timeframes: list[str]):
        df = candles_1m.sort_values("ts_min", kind="mergesort").drop_duplicates("ts_min")
        base = _bars_from_1m(df)
        self.spread_source_1m = (
            df["spread_source"].to_numpy(dtype=object) if "spread_source" in df else np.full(len(base), None, dtype=object)
        )
        self.series: dict[str, BarSeries] = {"1m": base}
        for tf in timeframes:
            if tf != "1m":
                self.series[tf] = resample(base, tf)

    def __getitem__(self, tf: str) -> BarSeries:
        return self.series[tf]

    @property
    def base(self) -> BarSeries:
        return self.series["1m"]

    def spread_source_at(self, as_of_min: int) -> str | None:
        idx = self.base.completed_index(as_of_min) - 1
        return None if idx < 0 else self.spread_source_1m[idx]

    def first_minute(self) -> int | None:
        return int(self.base.start[0]) if len(self.base) else None

    def last_minute(self) -> int | None:
        return int(self.base.start[-1]) if len(self.base) else None
