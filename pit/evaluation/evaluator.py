"""Forecast outcome evaluation.

Measures kept deliberately separate:

* **Target touch** (mid prices): did the mid price reach the target before expiry?
* **Target first**: was the target touched before the equidistant mirror target? (control)
* **Directional accuracy at expiry**: did the mid close at expiry lie beyond the reference
  price in the forecast direction?
* **Executable trade**: a hypothetical bracket order entered at the first candle's open
  (long at ASK, short at BID, plus slippage) with take-profit at the target and a stop at
  ``stop_price``; exits use the opposite side (long exits at BID, short at ASK).

Intrabar ordering: a 1-minute candle does not reveal whether its high or low came first.
If one candle satisfies both the stop and the target, the trade outcome is ``ambiguous`` -
it is never counted as a success - and its P&L assumes the stop was hit first
(conservative). If tick quotes covering that minute are stored, they resolve the order.
A candle that *opens* beyond a level is unambiguous: the open is its first price.

Coverage: a target *hit* is an observed fact. A *miss* is only declared when at least
``min_coverage`` of the expected open-market minutes have candles; otherwise the status is
``indeterminate``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pit.common.timeutils import iso, iso_min, utcnow
from pit.config.settings import EvaluationConfig
from pit.forecasting.models import Forecast, ForecastTarget


@dataclass
class CandleWindow:
    """1-minute candles (arrays aligned by index) for ``[start_min, expiry_min)``."""

    ts: np.ndarray
    mid_o: np.ndarray
    mid_h: np.ndarray
    mid_l: np.ndarray
    mid_c: np.ndarray
    bid_o: np.ndarray
    bid_h: np.ndarray
    bid_l: np.ndarray
    bid_c: np.ndarray
    ask_o: np.ndarray
    ask_h: np.ndarray
    ask_l: np.ndarray
    ask_c: np.ndarray

    def __len__(self) -> int:
        return int(self.ts.size)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "CandleWindow":
        g = lambda c: df[c].to_numpy(dtype="float64")  # noqa: E731
        return cls(
            ts=df["ts_min"].to_numpy(dtype="int64"),
            mid_o=g("open"), mid_h=g("high"), mid_l=g("low"), mid_c=g("close"),
            bid_o=g("bid_open"), bid_h=g("bid_high"), bid_l=g("bid_low"), bid_c=g("bid_close"),
            ask_o=g("ask_open"), ask_h=g("ask_high"), ask_l=g("ask_low"), ask_c=g("ask_close"),
        )

    def slice(self, lo: int, hi: int) -> "CandleWindow":
        return CandleWindow(*(getattr(self, f)[lo:hi] for f in self.__dataclass_fields__))


class CandleStore:
    """Sorted 1-minute candles supporting fast window extraction."""

    def __init__(self, df: pd.DataFrame):
        self.window = CandleWindow.from_frame(df.sort_values("ts_min", kind="mergesort"))

    def between(self, start_min: int, end_min: int) -> CandleWindow:
        lo = int(np.searchsorted(self.window.ts, start_min, side="left"))
        hi = int(np.searchsorted(self.window.ts, end_min, side="left"))
        return self.window.slice(lo, hi)

    @property
    def last_minute(self) -> int | None:
        return int(self.window.ts[-1]) if len(self.window) else None


# --------------------------------------------------------------------------- primitives
def first_touch(window: CandleWindow, direction: str, price: float) -> int | None:
    """Index of the first candle whose mid range reaches ``price`` in ``direction``."""
    if len(window) == 0:
        return None
    hits = window.mid_h >= price if direction == "UP" else window.mid_l <= price
    idx = np.flatnonzero(hits)
    return int(idx[0]) if idx.size else None


@dataclass
class TradeResult:
    outcome: str  # tp | sl | expired | ambiguous | not_filled | indeterminate
    entry_idx: int | None = None
    entry_price: float | None = None
    exit_idx: int | None = None
    exit_price: float | None = None
    pnl_per_unit: float | None = None
    ambiguous: bool = False
    resolved_by_ticks: bool = False


def resolve_with_ticks(ticks: pd.DataFrame | None, minute: int, direction: str, tp: float, sl: float) -> str | None:
    """Use stored tick quotes to decide whether TP or SL traded first within ``minute``."""
    if ticks is None or ticks.empty:
        return None
    t = ticks[(ticks["ts_ms"] >= minute * 60_000) & (ticks["ts_ms"] < (minute + 1) * 60_000)]
    for row in t.itertuples(index=False):
        if direction == "UP":
            if row.bid <= sl:
                return "sl"
            if row.bid >= tp:
                return "tp"
        else:
            if row.ask >= sl:
                return "sl"
            if row.ask <= tp:
                return "tp"
    return None


def simulate_bracket(
    window: CandleWindow,
    direction: str,
    take_profit: float,
    stop: float,
    slippage: float,
    ticks: pd.DataFrame | None = None,
) -> TradeResult:
    """Enter at the first candle's open, exit at TP/SL/expiry using the executable side."""
    if len(window) == 0:
        return TradeResult("not_filled")
    long = direction == "UP"
    entry = (window.ask_o[0] + slippage) if long else (window.bid_o[0] - slippage)
    sign = 1.0 if long else -1.0
    # Exit-side prices: a long position is closed by selling at the bid; a short by buying at the ask.
    ex_o, ex_h, ex_l, ex_c = (
        (window.bid_o, window.bid_h, window.bid_l, window.bid_c)
        if long else (window.ask_o, window.ask_h, window.ask_l, window.ask_c)
    )
    for i in range(len(window)):
        if long:
            sl_hit, tp_hit = ex_l[i] <= stop, ex_h[i] >= take_profit
            gap_sl, gap_tp = ex_o[i] <= stop, ex_o[i] >= take_profit
        else:
            sl_hit, tp_hit = ex_h[i] >= stop, ex_l[i] <= take_profit
            gap_sl, gap_tp = ex_o[i] >= stop, ex_o[i] <= take_profit
        if i == 0:
            # The entry happens at the open of candle 0, so the open cannot already be "beyond" a level
            # in a way that triggers before entry; treat the open as the entry price.
            gap_sl = gap_tp = False
        if not (sl_hit or tp_hit):
            continue
        sl_exit = (min(ex_o[i], stop) if long else max(ex_o[i], stop)) if gap_sl else stop
        sl_exit = sl_exit - slippage if long else sl_exit + slippage
        if gap_sl:
            return TradeResult("sl", 0, entry, i, sl_exit, sign * (sl_exit - entry))
        if gap_tp:
            return TradeResult("tp", 0, entry, i, take_profit, sign * (take_profit - entry))
        if sl_hit and tp_hit:
            order = resolve_with_ticks(ticks, int(window.ts[i]), direction, take_profit, stop)
            if order == "tp":
                return TradeResult("tp", 0, entry, i, take_profit, sign * (take_profit - entry), resolved_by_ticks=True)
            if order == "sl":
                return TradeResult("sl", 0, entry, i, sl_exit, sign * (sl_exit - entry), resolved_by_ticks=True)
            # Unknown order: never a success; P&L assumes the stop came first.
            return TradeResult("ambiguous", 0, entry, i, sl_exit, sign * (sl_exit - entry), ambiguous=True)
        if sl_hit:
            return TradeResult("sl", 0, entry, i, sl_exit, sign * (sl_exit - entry))
        return TradeResult("tp", 0, entry, i, take_profit, sign * (take_profit - entry))
    last = len(window) - 1
    exit_price = (ex_c[last] - slippage) if long else (ex_c[last] + slippage)
    return TradeResult("expired", 0, entry, last, exit_price, sign * (exit_price - entry))


# --------------------------------------------------------------------------- evaluation
def evaluate_forecast(
    f: Forecast,
    targets: list[ForecastTarget],
    window: CandleWindow,
    expected_minutes: int,
    cfg: EvaluationConfig,
    ticks: pd.DataFrame | None = None,
    evaluated_at: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated_at = evaluated_at or iso(utcnow())
    base = {
        "evaluation_id": uuid.uuid4().hex,
        "forecast_id": f.forecast_id,
        "evaluator_version": cfg.version,
        "evaluated_at": evaluated_at,
        "status": "not_applicable",
        "coverage": None,
        "n_candles": int(len(window)),
        "expected_minutes": int(expected_minutes),
        "first_touch_ts": None,
        "minutes_to_touch": None,
        "target_first": None,
        "mfe": None,
        "mae": None,
        "mae_before_hit": None,
        "mae_before_hit_ambiguous": None,
        "close_at_expiry": None,
        "direction_correct": None,
        "trade_outcome": "not_applicable",
        "trade_entry_ts": None,
        "trade_entry_price": None,
        "trade_exit_ts": None,
        "trade_exit_price": None,
        "trade_pnl_per_oz": None,
        "trade_ambiguous": None,
        "details_json": None,
    }
    direction = f.candidate_direction
    primary = next((t for t in targets if t.is_primary), None)
    if direction not in ("UP", "DOWN") or primary is None or f.reference_price is None:
        return base, []

    coverage = (len(window) / expected_minutes) if expected_minutes > 0 else 0.0
    coverage = min(1.0, coverage)
    enough = coverage >= cfg.min_coverage
    base["coverage"] = round(coverage, 4)
    ref = f.reference_price
    s = 1.0 if direction == "UP" else -1.0

    target_rows = []
    touch_idx: dict[str, int | None] = {}
    for t in targets:
        idx = first_touch(window, t.direction, t.price)
        touch_idx[t.label] = idx
        hit = 1 if idx is not None else (0 if enough else None)
        target_rows.append(
            {
                "forecast_id": f.forecast_id,
                "evaluator_version": cfg.version,
                "label": t.label,
                "hit": hit,
                "first_touch_ts": iso_min(int(window.ts[idx])) if idx is not None else None,
                "minutes_to_touch": int(window.ts[idx] - f.start_min + 1) if idx is not None else None,
            }
        )

    i_t = touch_idx.get("primary")
    i_m = touch_idx.get("mirror")
    if i_t is not None:
        base["status"] = "hit"
        base["first_touch_ts"] = iso_min(int(window.ts[i_t]))
        base["minutes_to_touch"] = int(window.ts[i_t] - f.start_min + 1)
    else:
        base["status"] = "miss" if enough else "indeterminate"

    if i_t is None and i_m is None:
        base["target_first"] = "neither" if enough else None
    elif i_m is None:
        base["target_first"] = "yes"
    elif i_t is None:
        base["target_first"] = "no"
    else:
        base["target_first"] = "yes" if i_t < i_m else ("no" if i_t > i_m else "ambiguous")

    if len(window):
        hi, lo = float(window.mid_h.max()), float(window.mid_l.min())
        fav, adv = (hi - ref, ref - lo) if s > 0 else (ref - lo, hi - ref)
        base["mfe"] = round(max(0.0, fav), 6)
        base["mae"] = round(max(0.0, adv), 6)
        close = float(window.mid_c[-1])
        base["close_at_expiry"] = close
        if enough:
            base["direction_correct"] = int(np.sign(close - ref) == s)
        if i_t is not None:
            adv_series = (ref - window.mid_l) if s > 0 else (window.mid_h - ref)
            before = float(adv_series[:i_t].max()) if i_t > 0 else 0.0
            at_hit = float(adv_series[i_t])
            base["mae_before_hit"] = round(max(0.0, before, at_hit), 6)
            base["mae_before_hit_ambiguous"] = int(at_hit > max(0.0, before))

    stop = f.stop_price
    if stop is not None:
        trade = simulate_bracket(window, direction, primary.price, stop, cfg.slippage, ticks)
        outcome = trade.outcome
        # Without enough data a non-triggered trade cannot be judged.
        if outcome == "expired" and not enough:
            outcome = "indeterminate"
        base.update(
            {
                "trade_outcome": outcome,
                "trade_entry_ts": iso_min(int(window.ts[trade.entry_idx])) if trade.entry_idx is not None else None,
                "trade_entry_price": trade.entry_price,
                "trade_exit_ts": iso_min(int(window.ts[trade.exit_idx])) if trade.exit_idx is not None else None,
                "trade_exit_price": trade.exit_price,
                "trade_pnl_per_oz": None if trade.pnl_per_unit is None or outcome == "indeterminate" else round(trade.pnl_per_unit, 6),
                "trade_ambiguous": int(trade.ambiguous),
            }
        )
        base["details_json"] = json.dumps({"resolved_by_ticks": trade.resolved_by_ticks,
                                           "entry_delay_min": int(window.ts[0] - f.start_min) if len(window) else None})
    return base, target_rows
