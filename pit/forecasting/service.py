"""Live forecasting service: load stored candles, generate, and persist slot forecasts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from pit.app.context import AppContext
from pit.common.timeutils import ensure_utc, iso_min, to_epoch_min, utcnow
from pit.data.market_calendar import MarketCalendar
from pit.features.frames import MarketFrames
from pit.forecasting.deterministic.engine import MODEL_TYPE, DeterministicForecaster
from pit.forecasting.models import ForecastContext, ForecastResult

log = logging.getLogger(__name__)


@dataclass
class SlotOutcome:
    slot_min: int
    status: str  # created | duplicate | market_closed | missed | no_data
    results: list[ForecastResult] = field(default_factory=list)
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "slot": iso_min(self.slot_min),
            "status": self.status,
            "message": self.message,
            "forecasts": [
                {
                    "forecast_id": r.forecast.forecast_id,
                    "model_type": r.forecast.model_type,
                    "horizon": r.forecast.horizon_minutes,
                    "direction": r.forecast.direction,
                    "candidate": r.forecast.candidate_direction,
                    "score": r.forecast.composite_score,
                    "target": r.forecast.target_price,
                    "p_hit": r.forecast.p_hit,
                    "confidence": r.forecast.confidence,
                    "regime": r.forecast.regime,
                    "wait_reasons": r.forecast.wait_reasons,
                    "data_mode": r.forecast.data_mode,
                }
                for r in self.results
            ],
        }


class ForecastService:
    def __init__(self, ctx: AppContext, source: str | None = None, llm_reviewer=None):
        self.ctx = ctx
        self.settings = ctx.settings
        self.calendar = MarketCalendar(ctx.settings.market)
        self.forecaster = DeterministicForecaster(ctx.settings, self.calendar)
        self.instrument = ctx.settings.instrument.symbol
        self.source = source or ctx.provider.name
        self.llm_reviewer = llm_reviewer

    def load_frames(self, as_of_min: int) -> MarketFrames:
        start = as_of_min - self.settings.data.history_days * 1440
        df = self.ctx.candles.load(self.instrument, self.source, start, as_of_min)
        return MarketFrames(df, self.settings.features.timeframes)

    def _context(self, as_of_min: int) -> ForecastContext:
        row = self.ctx.db.query_one(
            "SELECT data_mode, quote_convention FROM candles WHERE instrument = ? AND source = ? AND ts_min < ? "
            "ORDER BY ts_min DESC LIMIT 1",
            (self.instrument, self.source, as_of_min),
        )
        return ForecastContext(
            origin="live",
            data_mode=row["data_mode"] if row else getattr(self.ctx.provider, "data_mode", "unknown"),
            provider=self.source,
            quote_convention=row["quote_convention"] if row else getattr(self.ctx.provider, "quote_convention", "unknown"),
            instrument=self.instrument,
        )

    def run_slot(self, slot_min: int, now: datetime | None = None, enforce_grace: bool = True) -> SlotOutcome:
        """Create the forecast(s) for ``slot_min``. Never back-dates: a late slot is recorded as missed."""
        now = ensure_utc(now or utcnow())
        if not self.calendar.is_valid_forecast_slot(slot_min):
            return SlotOutcome(slot_min, "market_closed")
        version = self.forecaster.config_version
        if self.ctx.forecasts.slot_exists("live", "", self.instrument, slot_min, MODEL_TYPE, version):
            return SlotOutcome(slot_min, "duplicate")
        lateness = (now - (datetime.fromtimestamp(slot_min * 60, tz=now.tzinfo))).total_seconds()
        if lateness < 0:
            return SlotOutcome(slot_min, "not_due", message="slot is in the future")
        if enforce_grace and lateness > self.settings.scheduler.grace_seconds:
            self.ctx.ops.record_missed_slot(self.instrument, slot_min, iso_min(slot_min), f"late by {lateness:.0f}s")
            return SlotOutcome(slot_min, "missed", message=f"late by {lateness:.0f}s")

        frames = self.load_frames(slot_min)
        results = self.forecaster.forecast(frames, slot_min, self._context(slot_min), created_at=now)
        stored = []
        for r in results:
            if self.ctx.forecasts.insert(r, store_snapshot=True):
                stored.append(r)
        if self.llm_reviewer is not None:
            for r in list(stored):
                try:
                    reviewed = self.llm_reviewer.review(r, frames)
                except Exception as exc:  # the deterministic forecast must never depend on the LLM
                    log.exception("LLM review failed")
                    self.ctx.ops.log_api_error("llm", f"review failed: {exc}")
                    reviewed = None
                if reviewed is not None and self.ctx.forecasts.insert(reviewed, store_snapshot=False):
                    stored.append(reviewed)
        status = "created" if stored else "duplicate"
        for r in stored:
            log.info(
                "forecast %s %s h=%s dir=%s score=%s wait=%s", r.forecast.slot_ts, r.forecast.model_type,
                r.forecast.horizon_minutes, r.forecast.direction, r.forecast.composite_score, r.forecast.wait_reasons,
            )
        return SlotOutcome(slot_min, status, stored)

    def current_slot(self, now: datetime | None = None) -> int:
        now = ensure_utc(now or utcnow())
        m = to_epoch_min(now)
        interval = self.settings.forecasting.interval_minutes
        return m - (m % interval)

    def detect_missed_slots(self, now: datetime | None = None) -> list[int]:
        """Record valid slots in the lookback window that have no live forecast (never back-filled)."""
        now = ensure_utc(now or utcnow())
        interval = self.settings.forecasting.interval_minutes
        current = self.current_slot(now)
        lookback = self.settings.scheduler.missed_slot_lookback_hours * 60
        first_forecast = self.ctx.db.scalar("SELECT MIN(slot_min) FROM forecasts WHERE origin = 'live'")
        if first_forecast is None:
            return []
        start = max(int(first_forecast), current - lookback)
        existing = {
            r["slot_min"]
            for r in self.ctx.db.query(
                "SELECT DISTINCT slot_min FROM forecasts WHERE origin = 'live' AND slot_min >= ?", (start,)
            )
        }
        missed = []
        grace = self.settings.scheduler.grace_seconds
        slot = start - (start % interval)
        while slot <= current:
            overdue = now.timestamp() - slot * 60 > grace
            if overdue and slot >= start and slot not in existing and self.calendar.is_valid_forecast_slot(slot):
                if self.ctx.ops.record_missed_slot(self.instrument, slot, iso_min(slot), "no forecast recorded"):
                    missed.append(slot)
            slot += interval
        return missed
