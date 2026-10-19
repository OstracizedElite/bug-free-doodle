"""Paper-trading service: feeds stored forecasts and candles through the account engine."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from pit.app.context import AppContext
from pit.common.timeutils import ensure_utc, to_epoch_min, utcnow
from pit.data.market_calendar import MarketCalendar
from pit.evaluation.evaluator import CandleWindow
from pit.forecasting.models import Forecast
from pit.paper_trading.engine import PaperEvents, PaperTradingEngine

SIGNAL_MODEL = "deterministic"


def make_fx_lookup(fx_candles: pd.DataFrame, max_age_minutes: int) -> Callable[[int], float | None]:
    """AUD/USD close of the latest candle at or before ``minute`` (no older than ``max_age``)."""
    if fx_candles is None or fx_candles.empty:
        return lambda minute: None
    ts = fx_candles["ts_min"].to_numpy(dtype="int64")
    close = fx_candles["close"].to_numpy(dtype="float64")

    def lookup(minute: int) -> float | None:
        idx = int(np.searchsorted(ts, minute, side="right")) - 1
        if idx < 0 or minute - ts[idx] > max_age_minutes:
            return None
        return float(close[idx])

    return lookup


class PaperTradingService:
    def __init__(self, ctx: AppContext, source: str | None = None):
        self.ctx = ctx
        self.cfg = ctx.settings.paper
        self.calendar = MarketCalendar(ctx.settings.market)
        self.instrument = ctx.settings.instrument.symbol
        self.fx_symbol = ctx.settings.instrument.fx_symbol
        self.source = source or ctx.provider.name

    def account_id(self, origin: str = "live", run_id: str = "", create: bool = True) -> str | None:
        version = self.ctx.settings.paper_config_version
        row = self.ctx.paper.find_account(origin, version, run_id)
        if row:
            return row["account_id"]
        if not create:
            return None
        name = f"{origin} paper account ({self.cfg.base_currency}, config {version})"
        return self.ctx.paper.create_account(
            name, origin, run_id, version, self.ctx.settings.paper_config_dict(), self.cfg.base_currency,
            self.cfg.initial_balance,
        )

    def process(self, now: datetime | None = None) -> dict:
        """Process all completed candles since the last run for the live account."""
        if not self.cfg.enabled:
            return {"status": "disabled"}
        now = ensure_utc(now or utcnow())
        now_min = to_epoch_min(now)
        account_id = self.account_id("live")
        state = self.ctx.paper.load_state(account_id)
        start = state.last_min + 1 if state.last_min is not None else None
        if start is None:
            first = self.ctx.db.scalar(
                "SELECT MIN(start_min) FROM forecasts WHERE origin = 'live' AND model_type = ? AND direction != 'WAIT'",
                (SIGNAL_MODEL,),
            )
            if first is None:
                return {"status": "no_signals", "account_id": account_id}
            start = int(first)
        candles = self.ctx.candles.load(self.instrument, self.source, start, now_min)
        if candles.empty:
            return {"status": "no_new_candles", "account_id": account_id}
        rows = self.ctx.db.query(
            "SELECT forecast_id FROM forecasts WHERE origin = 'live' AND model_type = ? AND direction != 'WAIT' "
            "AND provider = ? AND start_min >= ? AND start_min < ?",
            (SIGNAL_MODEL, self.source, start, now_min),
        )
        forecasts = [self.ctx.forecasts.get(r["forecast_id"]) for r in rows]
        fx = self.ctx.candles.load(self.fx_symbol, self.source, start - self.cfg.fx_max_age_minutes, now_min)
        engine = PaperTradingEngine(
            self.cfg, self.calendar, account_id, make_fx_lookup(fx, self.cfg.fx_max_age_minutes), state,
            self.ctx.settings.instrument.price_decimals,
        )
        events = engine.process(CandleWindow.from_frame(candles), [f for f in forecasts if f is not None])
        self.ctx.paper.persist(account_id, engine.state, events)
        return {
            "status": "ok",
            "account_id": account_id,
            "opened": len(events.opened),
            "closed": len(events.closed),
            "transactions": len(events.transactions),
            "balance": round(engine.state.balance, 2),
            "open_positions": len(engine.state.open_positions),
        }


def run_paper_backtest(ctx: AppContext, run_id: str, candles: pd.DataFrame, fx_candles: pd.DataFrame,
                       forecasts: list[Forecast]) -> tuple[str, PaperEvents, PaperTradingEngine]:
    """Simulate the paper account over a backtest and persist it under ``origin='backtest'``."""
    cfg = ctx.settings.paper
    calendar = MarketCalendar(ctx.settings.market)
    account_id = ctx.paper.create_account(
        f"backtest {run_id}", "backtest", run_id, ctx.settings.paper_config_version,
        ctx.settings.paper_config_dict(), cfg.base_currency, cfg.initial_balance,
    )
    engine = PaperTradingEngine(
        cfg, calendar, account_id, make_fx_lookup(fx_candles, cfg.fx_max_age_minutes),
        decimals=ctx.settings.instrument.price_decimals,
    )
    signals = [f for f in forecasts if f.model_type == SIGNAL_MODEL]
    events = engine.process(CandleWindow.from_frame(candles), signals)
    ctx.paper.persist(account_id, engine.state, events)
    return account_id, events, engine
