"""Market-data domain types.

A *candle frame* is a ``pandas.DataFrame`` of 1-minute candles with the columns in
:data:`CANDLE_COLUMNS`. ``ts_min`` is the candle OPEN time in epoch minutes; a candle is
complete once ``ts_min + 1 <= now_min``. ``open/high/low/close`` are mid prices unless
``quote_convention`` states otherwise. Bid/ask columns are populated either from observed
provider data (``spread_source='observed'``), derived from a provider-supplied spread
(``'derived'``) or from the configured default spread (``'assumed'``). The source of every
spread is therefore always explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


class DataMode(str, Enum):
    SYNTHETIC = "synthetic"
    HISTORICAL = "historical"
    DELAYED = "delayed"
    LIVE = "live"


class SpreadSource(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    ASSUMED = "assumed"


MID_COLUMNS = ["open", "high", "low", "close"]
BID_COLUMNS = ["bid_open", "bid_high", "bid_low", "bid_close"]
ASK_COLUMNS = ["ask_open", "ask_high", "ask_low", "ask_close"]
PRICE_COLUMNS = MID_COLUMNS + BID_COLUMNS + ASK_COLUMNS
CANDLE_COLUMNS = (
    ["ts_min"]
    + PRICE_COLUMNS
    + ["spread", "spread_source", "volume", "quote_convention", "data_mode", "quality", "quality_notes"]
)


def empty_candle_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in CANDLE_COLUMNS})
    df["ts_min"] = df["ts_min"].astype("int64")
    for c in ("spread_source", "quote_convention", "data_mode", "quality", "quality_notes"):
        df[c] = df[c].astype("object")
    return df


def conform_candle_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exactly :data:`CANDLE_COLUMNS` (missing columns filled with NaN/None)."""
    out = df.copy()
    for c in CANDLE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan if c in PRICE_COLUMNS + ["spread", "volume"] else None
    out = out[CANDLE_COLUMNS]
    out["ts_min"] = out["ts_min"].astype("int64")
    for c in PRICE_COLUMNS + ["spread", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in ("spread_source", "quote_convention", "data_mode", "quality", "quality_notes"):
        out[c] = out[c].astype("object")
    return out.reset_index(drop=True)


def complete_bid_ask(df: pd.DataFrame, default_spread: float) -> pd.DataFrame:
    """Fill mid/bid/ask/spread consistently and label where each spread came from.

    Precedence per row:
      1. bid_close and ask_close present -> spread observed; missing bid/ask OHLC fields are
         derived from mid +/- spread/2; missing mid fields from (bid + ask) / 2.
      2. only a provider spread present -> bid/ask derived from mid +/- spread/2.
      3. nothing -> configured default spread, labelled 'assumed'.
    """
    out = conform_candle_frame(df)
    have_ba = out["bid_close"].notna() & out["ask_close"].notna()

    # Mid from bid/ask where mid is missing.
    for m, b, a in zip(MID_COLUMNS, BID_COLUMNS, ASK_COLUMNS):
        fill = out[m].isna() & out[b].notna() & out[a].notna()
        out.loc[fill, m] = (out.loc[fill, b] + out.loc[fill, a]) / 2.0

    observed_spread = out["ask_close"] - out["bid_close"]
    has_provider_spread = out["spread"].notna() & ~have_ba
    spread = np.where(have_ba, observed_spread, np.where(has_provider_spread, out["spread"], default_spread))
    source = np.where(
        have_ba,
        SpreadSource.OBSERVED.value,
        np.where(has_provider_spread, SpreadSource.DERIVED.value, SpreadSource.ASSUMED.value),
    )
    out["spread"] = spread.astype("float64")
    out["spread_source"] = source.astype(object)

    half = out["spread"] / 2.0
    for m, b, a in zip(MID_COLUMNS, BID_COLUMNS, ASK_COLUMNS):
        out[b] = out[b].where(out[b].notna(), out[m] - half)
        out[a] = out[a].where(out[a].notna(), out[m] + half)
    return out


class Quote(BaseModel):
    """A single bid/ask observation."""

    model_config = ConfigDict(frozen=True)

    instrument: str
    source: str
    ts: datetime
    bid: float
    ask: float
    data_mode: DataMode

    @model_validator(mode="after")
    def _check(self) -> "Quote":
        if not (self.bid > 0 and self.ask > 0):
            raise ValueError("bid and ask must be positive")
        if self.ask < self.bid:
            raise ValueError("ask must be >= bid")
        return self

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class ProviderCapabilities:
    name: str
    instruments: tuple[str, ...]
    historical_candles: bool
    live_quotes: bool
    bid_ask_candles: bool
    tick_quotes: bool
    streaming: bool
    requires_credentials: bool
    quote_convention: str
    notes: str = ""
