"""Historical candles from a local CSV file (e.g. an approved vendor export).

Accepted columns (case-insensitive; aliases in brackets):
  timestamp [ts, time, datetime, date, timestamp_utc]   required
  open, high, low, close                                required unless bid_*/ask_* given
  bid_open, bid_high, bid_low, bid_close                optional
  ask_open, ask_high, ask_low, ask_close                optional
  bid, ask                                              optional (treated as closing bid/ask)
  spread, volume                                        optional

Timestamps are candle OPEN times. Naive timestamps are interpreted in the configured
timezone (UTC by default). Rows must be 1-minute candles; anything else is rejected.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from pit.common.timeutils import ensure_utc, to_epoch_min
from pit.data.models import ProviderCapabilities, conform_candle_frame
from pit.data.providers.base import ProviderConfigurationError, MarketDataProvider
from pit.data.quality import timestamps_to_epoch_min

_TS_ALIASES = ("timestamp", "ts", "time", "datetime", "date", "timestamp_utc")


def read_candle_csv(path: str | Path, timezone: str = "UTC") -> tuple[pd.DataFrame, int]:
    """Parse a CSV into a candle frame. Returns ``(frame, misaligned_row_count)``."""
    path = Path(path)
    if not path.exists():
        raise ProviderConfigurationError(f"CSV file not found: {path}")
    raw = pd.read_csv(path, sep=None, engine="python")
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    ts_col = next((c for c in _TS_ALIASES if c in raw.columns), None)
    if ts_col is None:
        raise ProviderConfigurationError(f"CSV {path} has no timestamp column (expected one of {_TS_ALIASES})")
    if "bid" in raw.columns and "bid_close" not in raw.columns:
        raw["bid_close"] = raw["bid"]
    if "ask" in raw.columns and "ask_close" not in raw.columns:
        raw["ask_close"] = raw["ask"]
    has_mid = all(c in raw.columns for c in ("open", "high", "low", "close"))
    has_ba = all(c in raw.columns for c in ("bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"))
    if not (has_mid or has_ba):
        raise ProviderConfigurationError("CSV needs open/high/low/close or full bid_*/ask_* OHLC columns")
    minutes, aligned = timestamps_to_epoch_min(raw[ts_col], timezone=timezone)
    raw["ts_min"] = minutes
    misaligned = int((~aligned).sum())
    frame = conform_candle_frame(raw[aligned].drop(columns=[ts_col]))
    return frame, misaligned


class CsvProvider(MarketDataProvider):
    name = "csv"

    def __init__(
        self,
        path: str | Path,
        instrument: str = "XAU/USD",
        quote_convention: str = "provider_mid",
        data_mode: str = "historical",
        source_name: str = "csv",
        timezone: str = "UTC",
    ):
        self.path = Path(path)
        self.instrument = instrument
        self.quote_convention = quote_convention
        self.data_mode = data_mode
        self.name = source_name
        self.timezone = timezone
        self._frame: pd.DataFrame | None = None
        self.misaligned_rows = 0

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            instruments=(self.instrument,),
            historical_candles=True,
            live_quotes=False,
            bid_ask_candles=True,
            tick_quotes=False,
            streaming=False,
            requires_credentials=False,
            quote_convention=self.quote_convention,
            notes=f"Local CSV file {self.path}",
        )

    def _load(self) -> pd.DataFrame:
        if self._frame is None:
            frame, self.misaligned_rows = read_candle_csv(self.path, self.timezone)
            frame["quote_convention"] = self.quote_convention
            frame["data_mode"] = self.data_mode
            self._frame = frame.sort_values("ts_min", kind="mergesort").reset_index(drop=True)
        return self._frame

    def fetch_candles(self, instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
        if instrument != self.instrument:
            return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
        df = self._load()
        s, e = to_epoch_min(ensure_utc(start)), to_epoch_min(ensure_utc(end))
        return df[(df["ts_min"] >= s) & (df["ts_min"] < e)].reset_index(drop=True)

    def bounds(self) -> tuple[int, int] | None:
        df = self._load()
        if df.empty:
            return None
        return int(df["ts_min"].min()), int(df["ts_min"].max()) + 1
