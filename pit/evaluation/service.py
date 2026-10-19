"""Evaluate forecasts whose horizon has elapsed."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

import pandas as pd

from pit.app.context import AppContext
from pit.common.timeutils import ensure_utc, iso, to_epoch_min, utcnow
from pit.data.market_calendar import MarketCalendar
from pit.evaluation.evaluator import CandleStore, evaluate_forecast
from pit.forecasting.models import Forecast

log = logging.getLogger(__name__)

# If data never arrives (feed outage), evaluate with what exists after this long.
MAX_WAIT_MINUTES = 24 * 60


class EvaluationService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.cfg = ctx.settings.evaluation
        self.calendar = MarketCalendar(ctx.settings.market)

    def evaluate(self, forecasts: list[Forecast], store: CandleStore, evaluated_at: str | None = None,
                 ticks: pd.DataFrame | None = None) -> int:
        """Evaluate the given forecasts against an in-memory candle store and persist results."""
        if not forecasts:
            return 0
        targets = self.ctx.forecasts.targets_for([f.forecast_id for f in forecasts])
        evals, trows = [], []
        evaluated_at = evaluated_at or iso(utcnow())
        for f in forecasts:
            window = store.between(f.start_min, f.expiry_min)
            expected = self.calendar.count_open_minutes(f.start_min, f.expiry_min)
            e, t = evaluate_forecast(f, targets.get(f.forecast_id, []), window, expected, self.cfg, ticks, evaluated_at)
            evals.append(e)
            trows.extend(t)
        return self.ctx.forecasts.insert_evaluations(evals, trows)

    def evaluate_due(self, now: datetime | None = None, origin: str | None = "live", run_id: str | None = None) -> int:
        """Evaluate stored forecasts whose expiry is covered by stored data (or long overdue)."""
        now = ensure_utc(now or utcnow())
        now_min = to_epoch_min(now)
        pending = self.ctx.forecasts.pending_evaluation(self.cfg.version, now_min, origin=origin, run_id=run_id)
        if not pending:
            return 0
        groups: dict[tuple[str, str], list[Forecast]] = defaultdict(list)
        for f in pending:
            groups[(f.instrument, f.provider)].append(f)
        total = 0
        for (instrument, source), items in groups.items():
            latest = self.ctx.candles.latest_ts_min(instrument, source)
            data_until = (latest + 1) if latest is not None else None
            due = [
                f for f in items
                if (data_until is not None and f.expiry_min <= data_until) or f.expiry_min + MAX_WAIT_MINUTES <= now_min
            ]
            if not due:
                continue
            lo = min(f.start_min for f in due)
            hi = max(f.expiry_min for f in due)
            store = CandleStore(self.ctx.candles.load(instrument, source, lo, hi))
            ticks = self.ctx.candles.load_quotes(instrument, source, lo * 60_000, hi * 60_000)
            total += self.evaluate(due, store, iso(now), ticks if not ticks.empty else None)
        if total:
            log.info("evaluated %d forecasts", total)
        return total
