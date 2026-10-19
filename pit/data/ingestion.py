"""Market-data ingestion: fetch -> validate -> store -> detect gaps -> update feed status."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pit.app.context import AppContext
from pit.common.timeutils import ensure_utc, from_epoch_min, iso, iso_min, to_epoch_min, utcnow
from pit.data.market_calendar import MarketCalendar
from pit.data.providers.base import MarketDataProvider, ProviderError
from pit.data.quality import Gap, ValidationReport, detect_gaps, feed_state, validate_candles

log = logging.getLogger(__name__)

MAX_EVENTS_PER_BATCH = 20


@dataclass
class IngestResult:
    instrument: str
    source: str
    start: str | None = None
    end: str | None = None
    fetched: int = 0
    inserted: int = 0
    report: ValidationReport = field(default_factory=ValidationReport)
    gaps: list[Gap] = field(default_factory=list)
    feed_status: str = "unknown"
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "fetched": self.fetched,
            "inserted": self.inserted,
            "validation": self.report.as_dict(),
            "gaps": len(self.gaps),
            "missing_minutes": sum(g.missing_minutes for g in self.gaps),
            "feed_status": self.feed_status,
            "error": self.error,
        }


class IngestionService:
    def __init__(self, ctx: AppContext, provider: MarketDataProvider | None = None):
        self.ctx = ctx
        self.settings = ctx.settings
        self.provider = provider or ctx.provider
        self.calendar = MarketCalendar(ctx.settings.market)
        self.repo = ctx.candles
        self.ops = ctx.ops

    @property
    def source(self) -> str:
        return self.provider.name

    def instruments(self) -> list[str]:
        out = [self.settings.instrument.symbol]
        fx = self.settings.instrument.fx_symbol
        if self.settings.paper.base_currency != "USD" and self.provider.supports(fx):
            out.append(fx)
        return out

    def _sanity_for(self, instrument: str):
        data = self.settings.data
        if instrument == self.settings.instrument.fx_symbol:
            return data.fx_sanity, data.fx_default_spread
        return data.sanity, data.default_spread

    # ------------------------------------------------------------------ core batch step
    def _ingest_window(self, instrument: str, start_min: int, end_min: int, now_min: int, result: IngestResult) -> None:
        df = self.provider.fetch_candles(instrument, from_epoch_min(start_min), from_epoch_min(end_min))
        result.fetched += len(df)
        prev_close = self.repo.last_close_before(instrument, self.source, start_min)
        sanity, default_spread = self._sanity_for(instrument)
        clean, report = validate_candles(
            df,
            sanity,
            default_spread,
            prev_close=prev_close,
            now_min=now_min,
        )
        result.report.merge(report)
        result.inserted += self.repo.insert_candles(instrument, self.source, clean)
        self._log_report(instrument, report)

    def _log_report(self, instrument: str, report: ValidationReport) -> None:
        if report.duplicates:
            self.ops.log_quality_event(
                instrument,
                self.source,
                "duplicate",
                "warning" if report.conflicting_duplicates else "info",
                f"{report.duplicates} duplicate candles dropped ({report.conflicting_duplicates} conflicting)",
            )
        if report.misaligned:
            self.ops.log_quality_event(
                instrument, self.source, "rejected", "warning", f"{report.misaligned} misaligned timestamps rejected"
            )
        for ts_min, reason in report.rejected[:MAX_EVENTS_PER_BATCH]:
            self.ops.log_quality_event(instrument, self.source, "rejected", "error", reason, iso_min(ts_min))
        for ts_min, reason in report.suspect[:MAX_EVENTS_PER_BATCH]:
            self.ops.log_quality_event(instrument, self.source, "suspect", "warning", reason, iso_min(ts_min))

    def _record_gaps(self, instrument: str, start_min: int, end_min: int) -> list[Gap]:
        if end_min <= start_min:
            return []
        present = self.repo.existing_minutes(instrument, self.source, start_min, end_min)
        gaps = detect_gaps(present, self.calendar, start_min, end_min)
        new = self.repo.record_gaps(instrument, self.source, gaps)
        if new:
            total = sum(g.missing_minutes for g in gaps)
            self.ops.log_quality_event(
                instrument, self.source, "gap", "warning", f"{len(gaps)} gaps / {total} missing open minutes "
                f"between {iso_min(start_min)} and {iso_min(end_min)}"
            )
        return gaps

    # ------------------------------------------------------------------ historical backfill
    def backfill(self, start: datetime, end: datetime, instrument: str | None = None, now: datetime | None = None) -> IngestResult:
        instrument = instrument or self.settings.instrument.symbol
        now = ensure_utc(now or utcnow())
        now_min = to_epoch_min(now)
        s = to_epoch_min(ensure_utc(start))
        e = min(to_epoch_min(ensure_utc(end)), now_min)
        result = IngestResult(instrument=instrument, source=self.source, start=iso_min(s), end=iso_min(e))
        chunk = self.settings.data.backfill_chunk_minutes
        cursor = s
        try:
            while cursor < e:
                nxt = min(e, cursor + chunk)
                self._ingest_window(instrument, cursor, nxt, now_min, result)
                cursor = nxt
        except ProviderError as exc:
            result.error = str(exc)
            self.ops.log_quality_event(instrument, self.source, "provider_error", "error", str(exc))
            log.error("backfill failed for %s: %s", instrument, exc)
        first = self.repo.earliest_ts_min(instrument, self.source)
        last = self.repo.latest_ts_min(instrument, self.source)
        if first is not None and last is not None:
            result.gaps = self._record_gaps(instrument, max(s, first), min(cursor, last + 1))
        result.feed_status = self._update_feed(instrument, now, error=result.error)
        return result

    # ------------------------------------------------------------------ incremental sync
    def sync(self, now: datetime | None = None, instrument: str | None = None) -> IngestResult:
        """Fetch all completed candles since the last stored one."""
        instrument = instrument or self.settings.instrument.symbol
        now = ensure_utc(now or utcnow())
        now_min = to_epoch_min(now)
        last = self.repo.latest_ts_min(instrument, self.source)
        history_start = to_epoch_min(now - timedelta(days=self.settings.data.history_days))
        start_min = history_start if last is None else max(last + 1, history_start)
        result = IngestResult(instrument=instrument, source=self.source, start=iso_min(start_min), end=iso_min(now_min))
        if start_min >= now_min:
            result.feed_status = self._update_feed(instrument, now)
            return result
        try:
            cursor = start_min
            chunk = self.settings.data.backfill_chunk_minutes
            while cursor < now_min:
                nxt = min(now_min, cursor + chunk)
                self._ingest_window(instrument, cursor, nxt, now_min, result)
                cursor = nxt
        except ProviderError as exc:
            result.error = str(exc)
            log.warning("sync failed for %s: %s", instrument, exc)
        new_last = self.repo.latest_ts_min(instrument, self.source)
        if new_last is not None:
            # Gaps only between stored candles: the live edge may simply not have arrived yet.
            gap_start = start_min if last is None else last + 1
            result.gaps = self._record_gaps(instrument, gap_start, new_last)
        result.feed_status = self._update_feed(instrument, now, error=result.error)
        return result

    def sync_all(self, now: datetime | None = None) -> list[IngestResult]:
        return [self.sync(now, instrument=i) for i in self.instruments()]

    # ------------------------------------------------------------------ feed status
    def _update_feed(self, instrument: str, now: datetime, error: str | None = None) -> str:
        last = self.repo.latest_ts_min(instrument, self.source)
        status, age = feed_state(last, now, self.calendar, self.settings.data.stale_after_seconds)
        if error and status != "closed":
            status = "down" if (age is None or age > self.settings.data.stale_after_seconds) else status
        previous = next(
            (f for f in self.ops.feed_status() if f["instrument"] == instrument and f["source"] == self.source), None
        )
        prev_status = previous["status"] if previous else None
        if prev_status in ("stale", "down") and status == "ok":
            self.ops.log_quality_event(instrument, self.source, "recovered", "info", f"feed recovered from {prev_status}")
        elif status in ("stale", "down") and prev_status not in ("stale", "down"):
            self.ops.log_quality_event(
                instrument, self.source, "stale", "warning",
                f"feed {status}: latest candle age {age if age is None else round(age)}s",
            )
        self.ops.upsert_feed_status(
            instrument,
            self.source,
            status,
            data_mode=self.provider.data_mode,
            last_candle_ts=iso_min(last) if last is not None else None,
            success=error is None,
            error=error,
            latency_seconds=age,
            now=now,
        )
        return status
