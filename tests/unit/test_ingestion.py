from datetime import datetime, timedelta, timezone

import pandas as pd

from pit.common.timeutils import from_epoch_min, to_epoch_min
from pit.data.ingestion import IngestionService
from pit.data.market_calendar import MarketCalendar
from pit.data.models import ProviderCapabilities
from pit.data.providers.base import MarketDataProvider, ProviderUnavailableError
from pit.data.providers.synthetic import SyntheticProvider
from tests.helpers import MONDAY_10_UTC, make_candles

ANCHOR = {"providers": {"synthetic": {"anchor": "2026-01-01T00:00:00Z"}}}


def synthetic(ctx, now, **kw):
    cfg = ctx.settings.providers.synthetic.model_copy(update=kw)
    return SyntheticProvider(cfg, MarketCalendar(ctx.settings.market), clock=lambda: now)


def test_backfill_is_idempotent_and_records_gaps(make_ctx):
    ctx = make_ctx(ANCHOR)
    now = datetime(2026, 1, 8, 0, 0, tzinfo=timezone.utc)
    svc = IngestionService(ctx, provider=synthetic(ctx, now, missing_rate=0.02))
    start, end = datetime(2026, 1, 6, tzinfo=timezone.utc), datetime(2026, 1, 7, tzinfo=timezone.utc)
    r1 = svc.backfill(start, end, now=now)
    assert r1.error is None and r1.inserted > 1000
    assert r1.gaps, "dropped minutes must be reported as gaps"
    stored = ctx.db.scalar("SELECT COUNT(*) FROM data_gaps")
    assert stored == len(r1.gaps)
    r2 = svc.backfill(start, end, now=now)
    assert r2.inserted == 0
    assert ctx.db.scalar("SELECT COUNT(*) FROM candles") == r1.inserted
    assert ctx.db.scalar("SELECT COUNT(*) FROM data_gaps") == stored
    # Candles are labelled synthetic in the database.
    assert ctx.db.scalar("SELECT DISTINCT data_mode FROM candles") == "synthetic"


def test_sync_is_incremental_and_tracks_feed_status(make_ctx):
    ctx = make_ctx({**ANCHOR, "data": {"history_days": 2}})
    t1 = datetime(2026, 1, 6, 12, 0, 20, tzinfo=timezone.utc)
    clock = {"now": t1}
    prov = SyntheticProvider(
        ctx.settings.providers.synthetic, MarketCalendar(ctx.settings.market), clock=lambda: clock["now"]
    )
    svc = IngestionService(ctx, provider=prov)
    r = svc.sync(t1)
    assert r.feed_status == "ok"
    last = ctx.candles.latest_ts_min("XAU/USD", "synthetic")
    assert last == to_epoch_min("2026-01-06T11:59:00Z")

    clock["now"] = t1 + timedelta(minutes=5)
    r2 = svc.sync(clock["now"])
    assert r2.inserted == 5
    assert ctx.candles.latest_ts_min("XAU/USD", "synthetic") == last + 5


class FlakyProvider(MarketDataProvider):
    name = "flaky"
    data_mode = "live"
    quote_convention = "test_mid"

    def __init__(self):
        self.fail = False
        self.frame = make_candles([2000 + 0.1 * i for i in range(30)])

    @property
    def capabilities(self):
        return ProviderCapabilities("flaky", ("XAU/USD",), True, False, True, False, False, False, "test_mid")

    def fetch_candles(self, instrument, start, end):
        if self.fail:
            raise ProviderUnavailableError("connection refused")
        s, e = to_epoch_min(start), to_epoch_min(end)
        return self.frame[(self.frame.ts_min >= s) & (self.frame.ts_min < e)]


def test_outage_marks_feed_down_then_recovers(make_ctx):
    ctx = make_ctx({"data": {"history_days": 1, "stale_after_seconds": 120}})
    prov = FlakyProvider()
    svc = IngestionService(ctx, provider=prov)
    now = from_epoch_min(MONDAY_10_UTC + 10) + timedelta(seconds=5)
    assert svc.sync(now).feed_status == "ok"

    prov.fail = True
    later = now + timedelta(minutes=5)
    r = svc.sync(later)
    assert r.error and r.feed_status == "down"
    assert ctx.db.scalar("SELECT COUNT(*) FROM data_quality_events WHERE event_type = 'stale'") == 1

    prov.fail = False
    r = svc.sync(later + timedelta(seconds=10))
    assert r.feed_status == "ok"
    assert ctx.db.scalar("SELECT COUNT(*) FROM data_quality_events WHERE event_type = 'recovered'") == 1
    (status,) = ctx.ops.feed_status()
    assert status["status"] == "ok" and status["last_error"] == "connection refused"


def test_rejected_and_suspect_candles_logged(make_ctx):
    ctx = make_ctx()
    frame = make_candles([2000, 2001, 2300, 2301, 2302])
    frame.loc[3, "high"] = 1000.0  # invalid

    class P(FlakyProvider):
        def fetch_candles(self, instrument, start, end):
            return frame

    svc = IngestionService(ctx, provider=P())
    r = svc.backfill(from_epoch_min(MONDAY_10_UTC), from_epoch_min(MONDAY_10_UTC + 5),
                     now=from_epoch_min(MONDAY_10_UTC + 6))
    assert r.inserted == 4
    assert len(r.report.rejected) == 1
    events = pd.DataFrame(ctx.ops.recent_quality_events())
    assert {"rejected", "suspect"} <= set(events["event_type"])
    assert ctx.db.scalar("SELECT COUNT(*) FROM candles WHERE quality = 'suspect'") == 1


def test_fx_pair_uses_fx_sanity_bounds(make_ctx):
    """Regression: gold price bounds must not reject the AUD/USD conversion pair."""
    ctx = make_ctx(ANCHOR)
    now = datetime(2026, 1, 8, 0, 0, tzinfo=timezone.utc)
    svc = IngestionService(ctx, provider=synthetic(ctx, now))
    assert "AUD/USD" in svc.instruments()
    r = svc.backfill(datetime(2026, 1, 6, tzinfo=timezone.utc), datetime(2026, 1, 7, tzinfo=timezone.utc),
                     instrument="AUD/USD", now=now)
    assert r.inserted > 1000 and not r.report.rejected
