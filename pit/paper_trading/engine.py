"""Virtual trading account (research only - no brokerage connection exists).

Event order within each 1-minute candle ``m``:

1. overnight financing for positions held across a rollover in ``(previous candle, m]``;
2. new positions for forecasts starting at/before ``m`` (entry at this candle's open:
   long at ASK + slippage, short at BID - slippage);
3. margin close-out check at the candle's worst prices (long: bid low, short: ask high);
4. take-profit / stop-loss on the executable side (same rules, gap handling and
   conservative ambiguity treatment as the evaluator);
5. expiry exit at the candle close when the horizon ends;
6. mark-to-market, drawdown and daily-loss limits, equity snapshot.

Currency: P&L is earned in USD. With an AUD base currency, every conversion uses the
AUD/USD rate observed at that minute (within ``fx_max_age_minutes``). If no rate exists the
trade is either skipped (at entry) or its USD P&L is held as *unconverted* (at exit); an AUD
value is never invented.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np

from pit.common.timeutils import from_epoch_min, iso_min, to_epoch_min
from pit.config.settings import PaperConfig
from pit.data.market_calendar import MarketCalendar
from pit.evaluation.evaluator import CandleWindow
from pit.forecasting.models import Forecast

CONFIDENCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class Position:
    position_id: str
    forecast_id: str
    direction: str
    units: float
    entry_min: int
    entry_price: float
    stop_price: float
    target_price: float
    expiry_min: int
    margin_usd: float
    fx_rate_entry: float | None
    financing_usd: float = 0.0

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "UP" else -1.0


@dataclass
class AccountState:
    balance: float
    peak_equity: float
    day_start_equity: float
    day: str | None = None
    daily_halted: bool = False
    drawdown_halted: bool = False
    unconverted_usd: float = 0.0
    seq: int = 0
    last_min: int | None = None
    last_fx: float | None = None
    open_positions: list[Position] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, text: str) -> "AccountState":
        d = json.loads(text)
        d["open_positions"] = [Position(**p) for p in d.get("open_positions", [])]
        return cls(**d)

    @classmethod
    def initial(cls, balance: float) -> "AccountState":
        return cls(balance=balance, peak_equity=balance, day_start_equity=balance)


@dataclass
class PaperEvents:
    opened: list[dict[str, Any]] = field(default_factory=list)
    closed: list[dict[str, Any]] = field(default_factory=list)
    transactions: list[dict[str, Any]] = field(default_factory=list)
    equity: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: "PaperEvents") -> None:
        self.opened.extend(other.opened)
        self.closed.extend(other.closed)
        self.transactions.extend(other.transactions)
        self.equity.extend(other.equity)


class PaperTradingEngine:
    def __init__(
        self,
        cfg: PaperConfig,
        calendar: MarketCalendar,
        account_id: str,
        fx_lookup: Callable[[int], float | None] | None = None,
        state: AccountState | None = None,
        decimals: int = 2,
        equity_every_minutes: int = 15,
    ):
        self.cfg = cfg
        self.calendar = calendar
        self.account_id = account_id
        self.base_usd = cfg.base_currency == "USD"
        self.fx_lookup = (lambda m: 1.0) if self.base_usd else (fx_lookup or (lambda m: None))
        self.state = state or AccountState.initial(cfg.initial_balance)
        self.decimals = decimals
        self.equity_every = equity_every_minutes
        ro_h, ro_m = (int(x) for x in cfg.financing.rollover_time_ny.split(":"))
        self._rollover = (ro_h, ro_m)

    # ------------------------------------------------------------------ helpers
    def _txn(self, ev: PaperEvents, minute: int, kind: str, position_id: str | None = None,
             forecast_id: str | None = None, amount_usd: float | None = None, amount_base: float | None = None,
             details: dict | None = None) -> None:
        self.state.seq += 1
        ev.transactions.append(
            {
                "txn_id": f"{self.account_id}-{self.state.seq:08d}",
                "account_id": self.account_id,
                "seq": self.state.seq,
                "ts": iso_min(minute),
                "txn_type": kind,
                "position_id": position_id,
                "forecast_id": forecast_id,
                "amount_usd": None if amount_usd is None else round(amount_usd, 6),
                "amount_base": None if amount_base is None else round(amount_base, 6),
                "balance_after": round(self.state.balance, 6),
                "details_json": json.dumps(details or {}, sort_keys=True),
            }
        )

    def _fx(self, minute: int) -> float | None:
        rate = self.fx_lookup(minute)
        if rate is not None and rate > 0:
            self.state.last_fx = rate
        return rate

    def _valuation_fx(self, minute: int) -> float | None:
        rate = self._fx(minute)
        return rate if rate is not None else self.state.last_fx

    def _unrealised_usd(self, p: Position, w: CandleWindow, i: int, worst: bool = False) -> float:
        if p.direction == "UP":
            px = w.bid_l[i] if worst else w.bid_c[i]
        else:
            px = w.ask_h[i] if worst else w.ask_c[i]
        return p.sign * (px - p.entry_price) * p.units - p.financing_usd

    def equity(self, w: CandleWindow, i: int, fx: float | None, worst: bool = False) -> float:
        if not self.state.open_positions:
            return self.state.balance
        fx = fx or self.state.last_fx or 1.0
        u = sum(self._unrealised_usd(p, w, i, worst) for p in self.state.open_positions)
        return self.state.balance + u / fx

    def _rollovers_between(self, prev_min: int, cur_min: int) -> int:
        if cur_min <= prev_min:
            return 0
        tz = self.calendar.tz
        start = from_epoch_min(prev_min).astimezone(tz).date()
        end = from_epoch_min(cur_min).astimezone(tz).date()
        n = 0
        d = start
        while d <= end:
            ro = datetime(d.year, d.month, d.day, self._rollover[0], self._rollover[1], tzinfo=tz)
            ro_min = to_epoch_min(ro)
            if prev_min < ro_min <= cur_min:
                n += 1
            d += timedelta(days=1)
        return n

    # ------------------------------------------------------------------ entries
    def _eligible(self, f: Forecast) -> str | None:
        c = self.cfg
        if f.direction not in ("UP", "DOWN"):
            return "not_directional"
        if f.target_price is None or f.target_distance is None or f.reference_price is None:
            return "no_target"
        if CONFIDENCE_RANK.get(f.confidence, 0) < CONFIDENCE_RANK[c.min_confidence]:
            return "confidence_below_minimum"
        if f.session not in c.allowed_sessions:
            return "session_not_allowed"
        if self.state.drawdown_halted:
            return "drawdown_halt"
        if self.state.daily_halted:
            return "daily_loss_halt"
        if len(self.state.open_positions) >= c.max_concurrent_positions:
            return "max_concurrent_positions"
        return None

    def _open(self, f: Forecast, w: CandleWindow, i: int, ev: PaperEvents) -> None:
        c = self.cfg
        m = int(w.ts[i])
        reason = self._eligible(f)
        if reason == "not_directional":
            return
        if reason:
            self._txn(ev, m, "skip", forecast_id=f.forecast_id, details={"reason": reason})
            return
        fx = self._fx(m)
        if fx is None:
            self._txn(ev, m, "skip", forecast_id=f.forecast_id, details={"reason": "fx_unavailable"})
            return
        long = f.direction == "UP"
        entry = (w.ask_o[i] + c.slippage) if long else (w.bid_o[i] - c.slippage)
        stop = f.reference_price - (1 if long else -1) * c.stop_multiple * f.target_distance
        stop = round(stop, self.decimals)
        risk_per_unit = abs(entry - stop) + c.slippage + 2 * c.commission_per_unit
        if (long and stop >= entry) or (not long and stop <= entry):
            self._txn(ev, m, "skip", forecast_id=f.forecast_id, details={"reason": "stop_beyond_entry"})
            return
        equity_base = self.equity(w, i, fx)
        equity_usd = equity_base * fx
        risk_usd = equity_usd * c.risk_per_trade
        units = math.floor(risk_usd / risk_per_unit / c.units_step) * c.units_step
        max_units_notional = math.floor(c.max_notional_multiple * equity_usd / entry / c.units_step) * c.units_step
        used_margin = sum(p.margin_usd for p in self.state.open_positions)
        free_margin_usd = equity_usd - used_margin
        max_units_margin = math.floor(max(0.0, free_margin_usd) * c.leverage / entry / c.units_step) * c.units_step
        units = min(units, max_units_notional, max_units_margin)
        if units < c.min_units:
            self._txn(ev, m, "skip", forecast_id=f.forecast_id,
                      details={"reason": "size_below_minimum", "units": units})
            return
        margin_usd = units * entry / c.leverage
        pos = Position(
            position_id=f"{self.account_id}-{f.forecast_id}",
            forecast_id=f.forecast_id,
            direction=f.direction,
            units=units,
            entry_min=m,
            entry_price=round(entry, 6),
            stop_price=stop,
            target_price=f.target_price,
            expiry_min=f.expiry_min,
            margin_usd=round(margin_usd, 6),
            fx_rate_entry=fx if not self.base_usd else None,
        )
        self.state.open_positions.append(pos)
        ev.opened.append(
            {
                "position_id": pos.position_id,
                "account_id": self.account_id,
                "forecast_id": f.forecast_id,
                "direction": pos.direction,
                "units": units,
                "entry_ts": iso_min(m),
                "entry_price": pos.entry_price,
                "stop_price": stop,
                "target_price": pos.target_price,
                "expiry_ts": iso_min(pos.expiry_min),
                "margin_usd": pos.margin_usd,
                "fx_rate_entry": pos.fx_rate_entry,
                "status": "open",
            }
        )
        commission = c.commission_per_unit * units
        self._txn(ev, m, "open", pos.position_id, f.forecast_id, amount_usd=-commission if commission else 0.0,
                  details={"units": units, "entry": pos.entry_price, "stop": stop, "target": pos.target_price,
                           "risk_usd": round(risk_usd, 2), "spread": round(float(w.ask_o[i] - w.bid_o[i]), 6)})

    # ------------------------------------------------------------------ exits
    def _exit_check(self, p: Position, w: CandleWindow, i: int) -> tuple[str, float, bool] | None:
        c = self.cfg
        long = p.direction == "UP"
        o, h, l = (w.bid_o[i], w.bid_h[i], w.bid_l[i]) if long else (w.ask_o[i], w.ask_h[i], w.ask_l[i])
        entry_candle = int(w.ts[i]) == p.entry_min
        if long:
            sl_hit, tp_hit = l <= p.stop_price, h >= p.target_price
            gap_sl, gap_tp = o <= p.stop_price, o >= p.target_price
        else:
            sl_hit, tp_hit = h >= p.stop_price, l <= p.target_price
            gap_sl, gap_tp = o >= p.stop_price, o <= p.target_price
        if entry_candle:
            gap_sl = gap_tp = False
        if not (sl_hit or tp_hit):
            return None
        sl_px = (min(o, p.stop_price) if long else max(o, p.stop_price)) if gap_sl else p.stop_price
        sl_px = sl_px - c.slippage if long else sl_px + c.slippage
        if gap_sl:
            return "sl", sl_px, False
        if gap_tp:
            return "tp", p.target_price, False
        if sl_hit and tp_hit:
            return "ambiguous_sl", sl_px, True
        if sl_hit:
            return "sl", sl_px, False
        return "tp", p.target_price, False

    def _close(self, p: Position, minute: int, exit_price: float, reason: str, ambiguous: bool, ev: PaperEvents) -> None:
        c = self.cfg
        gross = p.sign * (exit_price - p.entry_price) * p.units
        commission = c.commission_per_unit * p.units * 2
        net = gross - commission - p.financing_usd
        fx = self._fx(minute)
        if fx is None:
            self.state.unconverted_usd += net
            net_base = None
        else:
            net_base = net / fx
            self.state.balance += net_base
        self.state.open_positions = [q for q in self.state.open_positions if q.position_id != p.position_id]
        ev.closed.append(
            {
                "position_id": p.position_id,
                "status": "closed",
                "exit_ts": iso_min(minute),
                "exit_price": round(exit_price, 6),
                "exit_reason": reason,
                "gross_pnl_usd": round(gross, 6),
                "costs_usd": round(commission, 6),
                "financing_usd": round(p.financing_usd, 6),
                "net_pnl_usd": round(net, 6),
                "net_pnl_base": None if net_base is None else round(net_base, 6),
                "fx_rate_exit": fx if not self.base_usd else None,
                "ambiguous": int(ambiguous),
            }
        )
        self._txn(ev, minute, "closeout" if reason == "closeout" else "close", p.position_id, p.forecast_id,
                  amount_usd=net, amount_base=net_base,
                  details={"reason": reason, "exit": round(exit_price, 6), "units": p.units,
                           "fx_unavailable": net_base is None})

    def _snapshot(self, ev: PaperEvents, w: CandleWindow, i: int, fx: float | None) -> None:
        eq = self.equity(w, i, fx)
        m = int(w.ts[i])
        dd = (self.state.peak_equity - eq) / self.state.peak_equity if self.state.peak_equity > 0 else 0.0
        margin_base = sum(p.margin_usd for p in self.state.open_positions) / (fx or self.state.last_fx or 1.0)
        ev.equity.append(
            {
                "account_id": self.account_id,
                "ts_min": m,
                "ts": iso_min(m),
                "balance": round(self.state.balance, 6),
                "unrealised": round(eq - self.state.balance, 6),
                "equity": round(eq, 6),
                "margin_used": round(margin_base, 6),
                "drawdown": round(max(0.0, dd), 6),
                "open_positions": len(self.state.open_positions),
            }
        )

    # ------------------------------------------------------------------ main loop
    def process(self, w: CandleWindow, forecasts: list[Forecast], until_min: int | None = None) -> PaperEvents:
        """Process candles after ``state.last_min`` (and before ``until_min``) with the given signals.

        ``forecasts`` may include already-processed ones; only those starting after
        ``state.last_min`` are considered.
        """
        ev = PaperEvents()
        c = self.cfg
        last = self.state.last_min
        lo = 0 if last is None else int(np.searchsorted(w.ts, last, side="right"))
        hi = len(w) if until_min is None else int(np.searchsorted(w.ts, until_min, side="left"))
        if lo >= hi:
            return ev
        pending = sorted(
            (f for f in forecasts if (last is None or f.start_min > last) and f.direction in ("UP", "DOWN")),
            key=lambda f: (f.start_min, f.horizon_minutes, f.forecast_id),
        )
        fi = 0
        i = lo
        prev_min = last
        while i < hi:
            m = int(w.ts[i])
            # Fast-forward while flat: nothing can happen until the next signal.
            if not self.state.open_positions:
                while fi < len(pending) and pending[fi].expiry_min <= m:
                    self._txn(ev, m, "skip", forecast_id=pending[fi].forecast_id, details={"reason": "no_candle_before_expiry"})
                    fi += 1
                if fi >= len(pending):
                    break
                nxt = pending[fi].start_min
                if m < nxt:
                    j = int(np.searchsorted(w.ts, nxt, side="left"))
                    if j >= hi:
                        break
                    i, prev_min = j, int(w.ts[j - 1]) if j > 0 else prev_min
                    m = int(w.ts[i])
            fx = self._valuation_fx(m)

            # New UTC day: reset the daily loss reference.
            day = from_epoch_min(m).strftime("%Y-%m-%d")
            if self.state.day != day:
                if self.state.daily_halted:
                    self._txn(ev, m, "resume", details={"reason": "new_day"})
                self.state.day = day
                self.state.daily_halted = False
                self.state.day_start_equity = self.equity(w, i, fx)

            # 1. financing across rollovers
            if self.state.open_positions and prev_min is not None:
                n_ro = self._rollovers_between(prev_min, m)
                if n_ro:
                    for p in self.state.open_positions:
                        rate = c.financing.long_annual_rate if p.direction == "UP" else c.financing.short_annual_rate
                        charge = p.units * p.entry_price * rate / 365.0 * n_ro
                        p.financing_usd += charge
                        self._txn(ev, m, "financing", p.position_id, p.forecast_id, amount_usd=-charge,
                                  details={"rollovers": n_ro, "annual_rate": rate})

            # 2. entries
            while fi < len(pending) and pending[fi].start_min <= m:
                f = pending[fi]
                fi += 1
                if f.expiry_min <= m:
                    self._txn(ev, m, "skip", forecast_id=f.forecast_id, details={"reason": "no_candle_before_expiry"})
                    continue
                before = len(ev.opened)
                self._open(f, w, i, ev)
                if len(ev.opened) > before:
                    self._snapshot(ev, w, i, fx)

            if self.state.open_positions:
                # 3. margin close-out at worst prices
                margin_used = sum(p.margin_usd for p in self.state.open_positions)
                fxv = fx or self.state.last_fx or 1.0
                eq_worst = self.equity(w, i, fx, worst=True)
                if eq_worst < c.margin_closeout_level * margin_used / fxv:
                    for p in list(self.state.open_positions):
                        px = w.bid_l[i] - c.slippage if p.direction == "UP" else w.ask_h[i] + c.slippage
                        self._close(p, m, float(px), "closeout", False, ev)
                # 4. TP / SL, 5. expiry
                for p in list(self.state.open_positions):
                    hit = self._exit_check(p, w, i)
                    if hit:
                        self._close(p, m, float(hit[1]), hit[0], hit[2], ev)
                    elif m + 1 >= p.expiry_min:
                        px = w.bid_c[i] - c.slippage if p.direction == "UP" else w.ask_c[i] + c.slippage
                        self._close(p, m, float(px), "expiry", False, ev)
                    else:
                        continue
                    self._snapshot(ev, w, i, fx)

            # 6. limits and periodic snapshot
            eq = self.equity(w, i, fx)
            self.state.peak_equity = max(self.state.peak_equity, eq)
            dd = (self.state.peak_equity - eq) / self.state.peak_equity if self.state.peak_equity else 0.0
            if not self.state.drawdown_halted and dd >= c.max_drawdown:
                for p in list(self.state.open_positions):
                    px = w.bid_c[i] - c.slippage if p.direction == "UP" else w.ask_c[i] + c.slippage
                    self._close(p, m, float(px), "drawdown_halt", False, ev)
                self.state.drawdown_halted = True
                self._txn(ev, m, "halt", details={"reason": "max_drawdown", "drawdown": round(dd, 6)})
                self._snapshot(ev, w, i, fx)
            day_loss = (self.state.day_start_equity - eq) / self.state.day_start_equity if self.state.day_start_equity else 0.0
            if not self.state.daily_halted and day_loss >= c.max_daily_loss:
                self.state.daily_halted = True
                self._txn(ev, m, "halt", details={"reason": "max_daily_loss", "day_loss": round(day_loss, 6)})
            if self.state.open_positions and m % self.equity_every == 0:
                self._snapshot(ev, w, i, fx)
            prev_min = m
            self.state.last_min = m
            i += 1
        # Every candle before ``hi`` is now accounted for (the loop only exits early while flat with
        # no signal due). Signals starting after the processed data remain pending for the next call.
        self.state.last_min = int(w.ts[hi - 1])
        return ev
