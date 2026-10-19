"""Scheduler behaviour under a simulated clock: unattended operation and recovery."""

from datetime import datetime, timedelta, timezone

import pytest

from pit.common.timeutils import to_epoch_min
from pit.data.ingestion import IngestionService
from pit.data.market_calendar import MarketCalendar
from pit.data.providers.base import ProviderUnavailableError
from pit.data.providers.synthetic import SyntheticProvider
from pit.scheduler.runner import Scheduler, SchedulerLockError

pytestmark = pytest.mark.integration

START = datetime(2026, 1, 14, 12, 0, 1, tzinfo=timezone.utc)  # Wednesday
OVERRIDES = {
    "providers": {"synthetic": {"anchor": "2025-12-01T00:00:00Z"}},
    "data": {"history_days": 30},
    "features": {"lookback_bars": {"4h": 60, "12h": 30}, "min_bars": 30},
}


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def env(make_ctx):
    ctx = make_ctx(OVERRIDES)
    clock = Clock(START)
    prov = SyntheticProvider(ctx.settings.providers.synthetic, MarketCalendar(ctx.settings.market), clock=clock)
    ctx.set_provider(prov)
    IngestionService(ctx).backfill(START - timedelta(days=30), START, now=START)
    return ctx, clock, prov


def run_for(sched: Scheduler, clock: Clock, minutes: int, step_seconds: int = 30) -> list:
    reports = []
    end = clock.t + timedelta(minutes=minutes)
    while clock.t < end:
        reports.append(sched.tick(clock.t))
        clock.t += timedelta(seconds=step_seconds)
    return reports


def live_slots(ctx):
    return [r["slot_min"] for r in ctx.db.query(
        "SELECT slot_min FROM forecasts WHERE origin = 'live' AND model_type = 'deterministic' ORDER BY slot_min")]


def test_unattended_operation_creates_each_slot_once_and_evaluates(env):
    ctx, clock, _ = env
    sched = Scheduler(ctx, clock=clock, use_lock=False, watch_config=False)
    sched.startup(clock.t)
    run_for(sched, clock, minutes=150)
    slots = live_slots(ctx)
    # Started at 12:00:01: the 12:00 slot is still inside the grace window, so it is forecast
    # (31 s late, stamped with its real creation time). Ticks end at 14:29:31 -> last slot 14:15.
    first = to_epoch_min("2026-01-14T12:00:00Z")
    assert slots == list(range(first, first + 15 * len(slots), 15))  # every slot, once, in order
    assert len(slots) == 10  # 12:00 .. 14:15
    # No forecast was created before its slot or outside the grace window.
    for r in ctx.db.query("SELECT slot_min, created_at FROM forecasts"):
        lag = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).timestamp() - r["slot_min"] * 60
        assert 0 <= lag <= ctx.settings.scheduler.grace_seconds
    assert ctx.db.scalar("SELECT COUNT(*) FROM forecast_evaluations") >= 6  # those already expired
    jobs = {r["job"] for r in ctx.ops.recent_jobs(500)}
    assert {"ingest", "forecast", "evaluate", "paper", "missed"} <= jobs
    assert ctx.ops.heartbeats()[0]["component"] == "scheduler"
    assert ctx.db.scalar("SELECT COUNT(*) FROM scheduler_runs WHERE status = 'error'") == 0


def test_restart_after_downtime_records_missed_slots_without_backfilling(env):
    ctx, clock, _ = env
    a = Scheduler(ctx, clock=clock, use_lock=False, watch_config=False)
    a.startup(clock.t)
    run_for(a, clock, minutes=40)  # forecasts 12:15, 12:30
    before = live_slots(ctx)
    clock.t += timedelta(minutes=80)  # process down from ~12:40 to ~14:00
    b = Scheduler(ctx, clock=clock, use_lock=False, watch_config=False)
    b.startup(clock.t)
    run_for(b, clock, minutes=20)
    after = live_slots(ctx)
    missed = {r["slot_min"] for r in ctx.ops.missed_slots()}
    gap = set(range(to_epoch_min("2026-01-14T12:45:00Z"), to_epoch_min("2026-01-14T14:00:00Z"), 15))
    assert gap <= missed
    assert not (gap & set(after))  # nothing back-filled for the outage
    assert set(before) <= set(after)
    assert to_epoch_min("2026-01-14T14:15:00Z") in after  # resumed normally


def test_provider_outage_is_isolated_and_forecasts_wait(env):
    ctx, clock, prov = env
    sched = Scheduler(ctx, clock=clock, use_lock=False, watch_config=False)
    sched.startup(clock.t)
    run_for(sched, clock, minutes=16)
    original = prov.fetch_candles

    def broken(*a, **k):
        raise ProviderUnavailableError("connection reset")

    prov.fetch_candles = broken
    run_for(sched, clock, minutes=30)  # slots 12:30 and 12:45 during the outage
    rows = ctx.db.query("SELECT slot_ts, direction, primary_wait_reason FROM forecasts ORDER BY slot_min")
    outage = [r for r in rows if r["slot_ts"] in ("2026-01-14T12:30:00Z", "2026-01-14T12:45:00Z")]
    assert outage and all(r["direction"] == "WAIT" and r["primary_wait_reason"] == "data_stale" for r in outage)
    status = {f["instrument"]: f["status"] for f in ctx.ops.feed_status()}
    assert status["XAU/USD"] in ("down", "stale")
    prov.fetch_candles = original
    run_for(sched, clock, minutes=20)
    assert {f["instrument"]: f["status"] for f in ctx.ops.feed_status()}["XAU/USD"] == "ok"
    last = ctx.db.query_one("SELECT primary_wait_reason FROM forecasts ORDER BY slot_min DESC LIMIT 1")
    assert last["primary_wait_reason"] != "data_stale"


def test_single_instance_lock(env):
    ctx, clock, _ = env
    a = Scheduler(ctx, clock=clock, use_lock=True, watch_config=False)
    a.startup(clock.t)
    b = Scheduler(ctx, clock=clock, use_lock=True, watch_config=False)
    with pytest.raises(SchedulerLockError):
        b.startup(clock.t)
    a.shutdown()
    b.startup(clock.t)  # lock released
    b.shutdown()


def test_weekend_produces_no_forecasts_and_no_missed_slots(env):
    ctx, clock, _ = env
    clock.t = datetime(2026, 1, 17, 10, 0, 1, tzinfo=timezone.utc)  # Saturday
    sched = Scheduler(ctx, clock=clock, use_lock=False, watch_config=False)
    sched.startup(clock.t)
    run_for(sched, clock, minutes=60, step_seconds=60)
    assert live_slots(ctx) == []
    weekend = [m for m in (r["slot_min"] for r in ctx.ops.missed_slots()) if m >= to_epoch_min("2026-01-16T22:00:00Z")]
    assert weekend == []
