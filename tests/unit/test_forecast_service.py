import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from pit.common.timeutils import from_epoch_min, to_epoch_min
from pit.data.ingestion import IngestionService
from pit.data.market_calendar import MarketCalendar
from pit.data.providers.synthetic import SyntheticProvider
from pit.forecasting.service import ForecastService

NOW = datetime(2026, 1, 14, 15, 0, 4, tzinfo=timezone.utc)  # Wednesday, market open
OVERRIDES = {
    "providers": {"synthetic": {"anchor": "2025-12-01T00:00:00Z"}},
    "data": {"history_days": 30},
    "features": {"lookback_bars": {"4h": 60, "12h": 30}, "min_bars": 30},
}


@pytest.fixture
def live(make_ctx):
    ctx = make_ctx(OVERRIDES)
    clock = {"now": NOW}
    prov = SyntheticProvider(ctx.settings.providers.synthetic, MarketCalendar(ctx.settings.market), clock=lambda: clock["now"])
    ctx.set_provider(prov)
    IngestionService(ctx).backfill(NOW - timedelta(days=30), NOW, now=NOW)
    return ctx, clock


def test_run_slot_creates_immutable_forecast_with_snapshot(live):
    ctx, _ = live
    svc = ForecastService(ctx)
    slot = to_epoch_min("2026-01-14T15:00:00Z")
    out = svc.run_slot(slot, NOW)
    assert out.status == "created" and len(out.results) == 1
    f = out.results[0].forecast
    assert f.data_mode == "synthetic" and f.origin == "live"
    assert f.start_min == slot + 1  # created 4 s after the slot -> evaluation starts next minute
    stored = ctx.forecasts.get(f.forecast_id)
    assert stored.content_hash == stored.computed_hash()
    assert ctx.forecasts.snapshot(f.snapshot_id)["features"]["input_hash"]
    if f.candidate_direction != "NONE":
        assert {t.label for t in ctx.forecasts.targets(f.forecast_id)} >= {"primary", "mirror"}
        with pytest.raises(sqlite3.DatabaseError):
            ctx.db.execute("UPDATE forecast_targets SET price = 0 WHERE forecast_id = ?", (f.forecast_id,))
    # Same slot again: duplicate prevented.
    assert svc.run_slot(slot, NOW + timedelta(seconds=5)).status == "duplicate"
    assert ctx.db.scalar("SELECT COUNT(*) FROM forecasts") == 1


def test_late_slot_is_recorded_as_missed_not_backdated(live):
    ctx, _ = live
    svc = ForecastService(ctx)
    slot = to_epoch_min("2026-01-14T14:30:00Z")
    out = svc.run_slot(slot, NOW)  # 30 minutes late
    assert out.status == "missed"
    assert ctx.db.scalar("SELECT COUNT(*) FROM forecasts") == 0
    assert ctx.ops.missed_slots()[0]["slot_min"] == slot


def test_closed_market_and_future_slots_skipped(live):
    ctx, _ = live
    svc = ForecastService(ctx)
    saturday = to_epoch_min("2026-01-17T12:00:00Z")
    assert svc.run_slot(saturday, from_epoch_min(saturday)).status == "market_closed"
    future = to_epoch_min("2026-01-14T15:15:00Z")
    assert svc.run_slot(future, NOW).status == "not_due"


def test_detect_missed_slots_after_downtime(live):
    ctx, clock = live
    svc = ForecastService(ctx)
    first = to_epoch_min("2026-01-14T15:00:00Z")
    svc.run_slot(first, NOW)
    later = NOW + timedelta(hours=1, minutes=10)
    missed = svc.detect_missed_slots(later)
    assert [from_epoch_min(m).strftime("%H:%M") for m in missed] == ["15:15", "15:30", "15:45", "16:00"]
    assert svc.detect_missed_slots(later) == []  # recorded once


def test_tampering_is_detectable(live):
    ctx, _ = live
    svc = ForecastService(ctx)
    out = svc.run_slot(to_epoch_min("2026-01-14T15:00:00Z"), NOW)
    fid = out.results[0].forecast.forecast_id
    assert ctx.forecasts.verify_hashes()["mismatched"] == []
    with ctx.db.connect() as conn:  # simulate someone bypassing the immutability trigger
        conn.execute("DROP TRIGGER trg_forecasts_no_update")
        conn.execute("UPDATE forecasts SET composite_score = 0.99 WHERE forecast_id = ?", (fid,))
    assert ctx.forecasts.verify_hashes()["mismatched"] == [fid]
