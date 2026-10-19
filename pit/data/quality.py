"""Market-data integrity checks.

Nothing here ever fabricates a price. Missing minutes are *recorded* as gaps; invalid
candles are *rejected*; implausible-but-possible candles are kept and flagged ``suspect``
so the forecasting engine can refuse to forecast on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from pit.common.timeutils import ensure_utc, iso_min
from pit.config.settings import SanityConfig
from pit.data.market_calendar import MarketCalendar
from pit.data.models import complete_bid_ask, conform_candle_frame

_EPS = 1e-9


@dataclass
class ValidationReport:
    received: int = 0
    accepted: int = 0
    duplicates: int = 0
    conflicting_duplicates: int = 0
    misaligned: int = 0
    incomplete: int = 0
    rejected: list[tuple[int, str]] = field(default_factory=list)
    suspect: list[tuple[int, str]] = field(default_factory=list)

    def merge(self, other: "ValidationReport") -> None:
        self.received += other.received
        self.accepted += other.accepted
        self.duplicates += other.duplicates
        self.conflicting_duplicates += other.conflicting_duplicates
        self.misaligned += other.misaligned
        self.incomplete += other.incomplete
        self.rejected.extend(other.rejected)
        self.suspect.extend(other.suspect)

    def as_dict(self) -> dict:
        return {
            "received": self.received,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "conflicting_duplicates": self.conflicting_duplicates,
            "misaligned": self.misaligned,
            "incomplete": self.incomplete,
            "rejected": len(self.rejected),
            "suspect": len(self.suspect),
        }


@dataclass(frozen=True)
class Gap:
    start_min: int  # first missing minute
    end_min: int  # exclusive
    missing_minutes: int

    def describe(self) -> str:
        return f"{iso_min(self.start_min)} .. {iso_min(self.end_min)} ({self.missing_minutes} open minutes missing)"


# ------------------------------------------------------------------------- timestamps
def timestamps_to_epoch_min(values: pd.Series, timezone: str = "UTC") -> tuple[np.ndarray, np.ndarray]:
    """Normalise timestamps of any supported form to epoch minutes.

    Accepts datetimes, ISO strings (with or without offset) and epoch seconds/milliseconds.
    Naive values are interpreted in ``timezone``. Returns ``(epoch_minutes, aligned_mask)``
    where ``aligned_mask`` is False for timestamps that are not on a whole minute.
    """
    if pd.api.types.is_numeric_dtype(values):
        arr = values.astype("float64").to_numpy()
        # Heuristic on magnitude: ms since epoch are > 1e11 for any date after 1973.
        seconds = np.where(arr > 1e11, arr / 1000.0, arr)
    else:
        parsed = _parse_datetimes(pd.Series(values).astype(str), timezone)
        # Robust to datetime64 unit (ns/us/ms) differences across pandas versions.
        seconds = (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    valid = np.isfinite(seconds)
    seconds = np.where(valid, seconds, 0.0)
    aligned = valid & np.isclose(np.mod(seconds, 60.0), 0.0, atol=1e-6)
    return np.floor(seconds / 60.0).astype("int64"), aligned


def _parse_datetimes(values: pd.Series, timezone: str) -> pd.Series:
    """Parse strings that may mix explicit offsets and naive local times.

    Naive values are localised to ``timezone``; DST-ambiguous or non-existent local times
    become NaT (and are subsequently rejected) rather than being guessed.
    """
    if timezone.upper() == "UTC":
        return pd.to_datetime(values, utc=True, format="mixed")
    has_offset = values.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if has_offset.any():
        out[has_offset] = pd.to_datetime(values[has_offset], utc=True, format="mixed")
    if (~has_offset).any():
        naive = pd.to_datetime(values[~has_offset], format="mixed")
        local = naive.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        out[~has_offset] = local.dt.tz_convert("UTC")
    return out


# ------------------------------------------------------------------------- validation
def validate_candles(
    df: pd.DataFrame,
    sanity: SanityConfig,
    default_spread: float,
    prev_close: float | None = None,
    now_min: int | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and normalise a provider candle batch.

    ``df`` must contain ``ts_min`` plus at least mid OHLC (or bid/ask OHLC). Returns the
    accepted candles sorted by time, with bid/ask/spread completed and ``quality`` set.
    """
    report = ValidationReport(received=int(len(df)))
    if df.empty:
        return conform_candle_frame(df), report

    work = conform_candle_frame(df)

    # Incomplete candles: open time + 1 minute must not be in the future.
    if now_min is not None:
        incomplete = work["ts_min"] + 1 > now_min
        report.incomplete = int(incomplete.sum())
        work = work[~incomplete]

    work = complete_bid_ask(work, default_spread)
    work = work.sort_values("ts_min", kind="mergesort")

    # Duplicates: identical rows are dropped silently; conflicting ones keep the first.
    dup_mask = work.duplicated("ts_min", keep="first")
    if dup_mask.any():
        report.duplicates = int(dup_mask.sum())
        dups = work[work["ts_min"].isin(work.loc[dup_mask, "ts_min"])]
        price_cols = ["open", "high", "low", "close"]
        conflicting = dups.groupby("ts_min")[price_cols].nunique().gt(1).any(axis=1)
        report.conflicting_duplicates = int(conflicting.sum())
        work = work[~dup_mask]

    o, h, l, c = (work[k].to_numpy() for k in ("open", "high", "low", "close"))
    bo, bh, bl, bc = (work[k].to_numpy() for k in ("bid_open", "bid_high", "bid_low", "bid_close"))
    ao, ah, al, ac = (work[k].to_numpy() for k in ("ask_open", "ask_high", "ask_low", "ask_close"))
    spread = work["spread"].to_numpy()
    ts = work["ts_min"].to_numpy()

    reasons = np.full(len(work), "", dtype=object)

    def reject(mask: np.ndarray, reason: str) -> None:
        idx = mask & (reasons == "")
        reasons[idx] = reason

    finite = np.isfinite(np.vstack([o, h, l, c, bc, ac])).all(axis=0)
    reject(~finite, "non_finite_price")
    with np.errstate(invalid="ignore"):
        reject((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0), "non_positive_price")
        reject(h + _EPS < np.maximum(o, c), "high_below_open_close")
        reject(l - _EPS > np.minimum(o, c), "low_above_open_close")
        reject(h + _EPS < l, "high_below_low")
        reject((c < sanity.min_price) | (c > sanity.max_price), "price_out_of_range")
        reject((bc - _EPS > ac) | (bo - _EPS > ao) | (bl - _EPS > al) | (bh - _EPS > ah), "bid_above_ask")
        reject(spread < -_EPS, "negative_spread")

    rejected = reasons != ""
    for t, r in zip(ts[rejected], reasons[rejected]):
        report.rejected.append((int(t), str(r)))

    keep = ~rejected
    work = work[keep].copy()
    if work.empty:
        return work.reset_index(drop=True), report

    # Suspect checks (kept, but flagged).
    closes = work["close"].to_numpy()
    prev = np.concatenate([[prev_close if prev_close else np.nan], closes[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        jump = np.abs(np.log(closes / prev))
    jump_flag = np.nan_to_num(jump, nan=0.0) > sanity.max_abs_1m_return
    wide_flag = (work["spread"].to_numpy() / closes) > sanity.max_spread_fraction
    notes = np.full(len(work), None, dtype=object)
    notes[jump_flag] = "price_jump"
    notes[wide_flag & ~jump_flag] = "wide_spread"
    notes[wide_flag & jump_flag] = "price_jump,wide_spread"
    suspect = jump_flag | wide_flag
    work["quality"] = np.where(suspect, "suspect", "ok").astype(object)
    work["quality_notes"] = notes
    for t, n in zip(work["ts_min"].to_numpy()[suspect], notes[suspect]):
        report.suspect.append((int(t), str(n)))

    report.accepted = int(len(work))
    return work.reset_index(drop=True), report


# ------------------------------------------------------------------------- gaps and staleness
def detect_gaps(present_minutes: np.ndarray, calendar: MarketCalendar, start_min: int, end_min: int) -> list[Gap]:
    """Runs of open-market minutes in ``[start_min, end_min)`` that have no candle."""
    expected = calendar.open_minutes(start_min, end_min)
    if expected.size == 0:
        return []
    missing = np.setdiff1d(expected, np.asarray(present_minutes, dtype="int64"), assume_unique=False)
    if missing.size == 0:
        return []
    # Split into runs of consecutive expected-open minutes. Two missing minutes belong to the
    # same run when no *present* open minute lies between them.
    positions = np.searchsorted(expected, missing)
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    gaps: list[Gap] = []
    for run in np.split(np.arange(missing.size), breaks):
        first, last = int(missing[run[0]]), int(missing[run[-1]])
        gaps.append(Gap(start_min=first, end_min=last + 1, missing_minutes=int(run.size)))
    return gaps


def feed_state(
    last_candle_min: int | None,
    now: datetime,
    calendar: MarketCalendar,
    stale_after_seconds: int,
) -> tuple[str, float | None]:
    """Classify the feed as ``ok``, ``stale``, ``closed`` or ``unknown``.

    Returns ``(status, age_seconds)`` where age is measured from the moment the latest
    candle *completed* (open time + 60 s).
    """
    now = ensure_utc(now)
    now_min = int(now.timestamp() // 60)
    market_open = calendar.is_open(now_min) or calendar.is_open(now_min - 1)
    if last_candle_min is None:
        return ("unknown" if market_open else "closed"), None
    age = now.timestamp() - (last_candle_min + 1) * 60
    if not market_open:
        return "closed", age
    return ("stale" if age > stale_after_seconds else "ok"), age
