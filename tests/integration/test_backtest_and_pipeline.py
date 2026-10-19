"""End-to-end: synthetic data -> ingestion -> replay -> evaluation -> paper trading -> analytics."""

from datetime import datetime, timedelta, timezone

import pytest

from pit.backtesting.replay import BacktestRunner, BacktestValidationError
from pit.common.timeutils import from_epoch_min, to_epoch_min
from pit.data.ingestion import IngestionService
from pit.data.market_calendar import MarketCalendar
from pit.data.providers.synthetic import SyntheticProvider
from pit.evaluation.service import EvaluationService
from pit.forecasting.service import ForecastService
from pit.paper_trading.service import PaperTradingService

pytestmark = pytest.mark.integration

NOW = datetime(2026, 2, 13, 0, 0, tzinfo=timezone.utc)
OVERRIDES = {
    "providers": {"synthetic": {"anchor": "2025-12-15T00:00:00Z"}},
    "data": {"history_days": 30},
    "features": {"lookback_bars": {"4h": 60, "12h": 30}, "min_bars": 30},
}


@pytest.fixture
def loaded(make_ctx):
    ctx = make_ctx(OVERRIDES)
    clock = {"now": NOW}
    ctx.set_provider(SyntheticProvider(ctx.settings.providers.synthetic, MarketCalendar(ctx.settings.market),
                                       clock=lambda: clock["now"]))
    svc = IngestionService(ctx)
    for inst in svc.instruments():
        svc.backfill(NOW - timedelta(days=50), NOW, instrument=inst, now=NOW)
    return ctx, clock


def test_backtest_end_to_end(loaded):
    ctx, _ = loaded
    r = BacktestRunner(ctx).run("2026-02-02T00:00:00Z", "2026-02-07T00:00:00Z", name="e2e")
    run = ctx.research.run(r.run_id)
    assert run["status"] == "completed" and run["validation_type"] == "exploratory"
    assert r.forecasts == r.summary["slots"] > 300
    assert r.evaluated == r.forecasts
    det = r.summary["deterministic"]
    assert det["n_forecasts"] == r.forecasts
    assert det["n_directional"] + det["n_wait"] == det["n_forecasts"]
    assert r.summary["warning"].startswith("SYNTHETIC")
    df = ctx.forecasts.frame(origin="backtest", run_id=r.run_id)
    assert (df["data_mode"] == "synthetic").all()
    assert df["outcome_status"].notna().all()
    # Paper account exists and is separate from forecast records.
    acct = ctx.paper.get_account(r.paper_account_id)
    assert acct["origin"] == "backtest" and acct["run_id"] == r.run_id
    trades = ctx.paper.positions(r.paper_account_id)
    assert set(trades["status"]) <= {"closed", "open"}
    assert ctx.forecasts.verify_hashes()["mismatched"] == []


def test_replay_forecast_equals_live_forecast(loaded):
    """The live service and the replay produce identical decisions for the same data."""
    ctx, _ = loaded
    slot = to_epoch_min("2026-02-04T14:00:00Z")
    r = BacktestRunner(ctx).run(from_epoch_min(slot), from_epoch_min(slot + 1), name="one-slot", paper=False)
    bt = ctx.forecasts.frame(origin="backtest", run_id=r.run_id).iloc[0]
    live = ForecastService(ctx).run_slot(slot, from_epoch_min(slot), enforce_grace=False).results[0].forecast
    for col in ("direction", "candidate_direction", "composite_score", "target_price", "p_hit", "regime",
                "snapshot_id", "wait_reasons_json", "reference_price", "stop_price"):
        assert bt[col] == getattr(live, col), col


def test_poisoned_future_does_not_leak_into_forecasts(loaded, make_ctx):
    """Corrupting every candle after T must not change the forecast made at T."""
    ctx, _ = loaded
    slot = to_epoch_min("2026-02-04T14:00:00Z")
    base = ctx.candles.load("XAU/USD", "synthetic", slot - 30 * 1440, slot + 600)
    poisoned = base.copy()
    future = poisoned["ts_min"] >= slot
    for col in ("open", "high", "low", "close", "bid_open", "bid_high", "bid_low", "bid_close",
                "ask_open", "ask_high", "ask_low", "ask_close"):
        poisoned.loc[future, col] = poisoned.loc[future, col] * 1.5
    from pit.features.frames import MarketFrames
    from pit.forecasting.deterministic.engine import DeterministicForecaster
    from pit.forecasting.models import ForecastContext

    fc = DeterministicForecaster(ctx.settings)
    fctx = ForecastContext("backtest", "synthetic", "synthetic", "synthetic_mid", run_id="x")
    a = fc.forecast(MarketFrames(base, ctx.settings.features.timeframes), slot, fctx, from_epoch_min(slot))[0]
    b = fc.forecast(MarketFrames(poisoned, ctx.settings.features.timeframes), slot, fctx, from_epoch_min(slot))[0]
    assert a.forecast.with_hash().content_hash == b.forecast.with_hash().content_hash


def test_out_of_sample_requires_unconsumed_matching_declaration(loaded):
    ctx, _ = loaded
    runner = BacktestRunner(ctx)
    with pytest.raises(BacktestValidationError):
        runner.run("2026-02-09T00:00:00Z", "2026-02-10T00:00:00Z", validation_type="out_of_sample")
    wrong = ctx.research.declare("deadbeef0000", "2026-02-09T00:00:00Z", "2026-02-12T00:00:00Z", "h")
    with pytest.raises(BacktestValidationError, match="differs"):
        runner.run("2026-02-09T00:00:00Z", "2026-02-10T00:00:00Z", validation_type="out_of_sample", declaration_id=wrong)
    decl = ctx.research.declare(ctx.settings.forecast_config_version, "2026-02-09T00:00:00Z",
                                "2026-02-12T00:00:00Z", "score >= 0.35 beats mirror after costs")
    with pytest.raises(BacktestValidationError, match="outside"):
        runner.run("2026-02-08T00:00:00Z", "2026-02-10T00:00:00Z", validation_type="out_of_sample", declaration_id=decl)
    r = runner.run("2026-02-09T00:00:00Z", "2026-02-10T00:00:00Z", validation_type="out_of_sample",
                   declaration_id=decl, paper=False)
    assert ctx.research.run(r.run_id)["validation_type"] == "out_of_sample"
    with pytest.raises(BacktestValidationError, match="already used"):
        runner.run("2026-02-09T00:00:00Z", "2026-02-10T00:00:00Z", validation_type="out_of_sample", declaration_id=decl)


def test_live_cycle_forecast_evaluate_paper(loaded):
    """Simulate a live afternoon: forecasts every 15 min, then evaluation and paper trading."""
    ctx, clock = loaded
    fsvc = ForecastService(ctx)
    start = to_epoch_min("2026-02-11T12:00:00Z")
    for k in range(12):
        slot = start + 15 * k
        assert fsvc.run_slot(slot, from_epoch_min(slot) + timedelta(seconds=3)).status == "created"
    later = from_epoch_min(start + 15 * 12 + 120)
    clock["now"] = later
    IngestionService(ctx).sync_all(later)
    n = EvaluationService(ctx).evaluate_due(later)
    assert n == 12
    assert EvaluationService(ctx).evaluate_due(later) == 0  # append-only, evaluated once
    out = PaperTradingService(ctx).process(later)
    assert out["status"] in ("ok", "no_signals")
    df = ctx.forecasts.frame(origin="live")
    assert df["outcome_status"].notna().all()
