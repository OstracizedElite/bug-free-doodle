import json
import math
from datetime import timedelta

import numpy as np
import pytest

from pit.common.timeutils import from_epoch_min, to_epoch_min
from pit.config import load_settings
from pit.data.market_calendar import MarketCalendar
from pit.features.engine import FeatureSnapshot
from pit.features.frames import MarketFrames
from pit.forecasting.deterministic import probability as prob
from pit.forecasting.deterministic.engine import DeterministicForecaster
from pit.forecasting.deterministic.scoring import (
    alignment_for_timeframe,
    composite_score,
    momentum_component,
    trend_component,
)
from pit.forecasting.deterministic.targets import build_targets, stop_price, target_distance
from pit.forecasting.events import EconomicEvent, EventCalendar
from pit.forecasting.models import ForecastContext
from tests.helpers import make_candles
from tests.unit.test_features import MON_00, NO_BREAK, small_settings

CTX = ForecastContext(origin="backtest", data_mode="historical", provider="test", quote_convention="test_mid", run_id="r1")


# ----------------------------------------------------------------------------- components
def test_component_normalisations():
    assert momentum_component(2.0, 2.0) == pytest.approx(math.tanh(1.0))
    assert momentum_component(None, 2.0) is None
    assert trend_component(1.0, 0.2, 1.0, 0.2) == pytest.approx(math.tanh(1.0))
    assert trend_component(-1.0, -0.2, 1.0, 0.2) == pytest.approx(-math.tanh(1.0))
    assert alignment_for_timeframe(10, 9, 8, 7) == 1.0
    assert alignment_for_timeframe(6, 7, 8, 9) == -1.0
    assert alignment_for_timeframe(10, 9, 8, 9) == pytest.approx(1 / 3)


def test_composite_score_weights_and_conflict(settings):
    fc = settings.forecasting
    comps = {"m1": 1.0, "m5": 1.0, "m15": 1.0, "trend60": 1.0, "ma_alignment": 1.0, "structure": 1.0, "vol_consistency": 1.0}
    assert composite_score(comps, fc).score == pytest.approx(1.0)
    comps = {"m1": -1.0, "m5": 0.5, "m15": 0.5, "trend60": 0.5, "ma_alignment": 0.0, "structure": -0.05, "vol_consistency": 0.2}
    r = composite_score(comps, fc)
    expected = 0.10 * -1 + 0.20 * 0.5 + 0.25 * 0.5 + 0.20 * 0.5 + 0 + 0.10 * -0.05 + 0.05 * 0.2
    assert r.score == pytest.approx(expected)
    # Only m1 opposes with magnitude >= 0.1 (structure -0.05 is below the conflict magnitude).
    assert r.conflict == pytest.approx(0.10)
    incomplete = dict(comps, structure=None)
    assert composite_score(incomplete, fc).score is None


# ----------------------------------------------------------------------------- targets / probability
def test_target_distance_uses_spread_floor(settings):
    t = settings.forecasting.targets
    assert target_distance(2.0, 0.3, t) == pytest.approx(1.5)  # 0.75 * ATR
    assert target_distance(0.4, 0.3, t) == pytest.approx(0.6)  # 2 * spread dominates
    levels = build_targets("DOWN", 2000.0, 2.0, 0.3, t)
    by = {lv.label: lv for lv in levels}
    assert by["primary"].price == 1998.5 and by["primary"].direction == "DOWN"
    assert by["mirror"].price == 2001.5 and by["mirror"].direction == "UP"
    assert by["atr_2"].price == 1996.0 and by["atr_0.5"].distance == pytest.approx(1.0)
    assert stop_price("DOWN", 2000.0, 1.5, 1.0) == 2001.5


def test_barrier_probability_and_heuristic_monotonicity(settings):
    # P(max W >= D) with D = sigma*sqrt(H) is 2*(1-Phi(1)) = 0.3173
    assert prob.barrier_touch_probability(1.0, 1.0, 1.0) == pytest.approx(0.31731, abs=1e-4)
    assert prob.barrier_touch_probability(0.0, 1.0, 1.0) == 1.0
    cfg = settings.forecasting.probability
    kw = dict(direction_sign=1, ma_alignment=0.5, distance=1.5, spread=0.3, sigma_1m=0.0003, reference=2000.0,
              horizon_minutes=60, cfg=cfg)
    p_weak, _ = prob.heuristic_p_hit(score=0.36, **kw)
    p_strong, d = prob.heuristic_p_hit(score=0.9, **kw)
    assert p_strong > p_weak
    assert cfg.min <= p_weak <= cfg.max
    far, _ = prob.heuristic_p_hit(score=0.9, **dict(kw, distance=10.0))
    assert far < p_strong
    assert "phit-heuristic-v1" in prob.formula_text(cfg)


# ----------------------------------------------------------------------------- full engine
def trending_frames(drift=0.08, n=4000, seed=2, spread=0.05, noise=0.3):
    rng = np.random.default_rng(seed)
    closes = 2000 + np.cumsum(drift + noise * rng.standard_normal(n))
    df = make_candles(closes, start_min=MON_00, spread=spread)
    return df, MarketFrames(df, ["1m", "5m", "15m", "30m", "60m", "4h", "12h"])


def test_engine_produces_up_forecast_on_strong_uptrend():
    s = small_settings(forecasting={"wait": {"max_conflict": 1.0, "block_session_close": False}})
    _, frames = trending_frames(drift=0.1, noise=0.1)
    eng = DeterministicForecaster(s, events=EventCalendar([]))
    slot = MON_00 + 3900
    (res,) = eng.forecast(frames, slot, CTX, created_at=from_epoch_min(slot))
    f = res.forecast
    assert f.composite_score > 0.35
    assert f.direction == "UP" and f.candidate_direction == "UP"
    assert f.target_price == pytest.approx(round(f.reference_price + f.target_distance, 2))
    assert f.stop_price < f.reference_price
    assert f.p_hit_method == "uncalibrated_heuristic"
    assert f.confidence in ("low", "medium", "high")
    assert f.start_min == slot and f.expiry_min == slot + 60
    assert {t.label for t in res.targets} == {"primary", "mirror", "atr_0.5", "atr_1", "atr_1.5", "atr_2"}
    assert json.loads(f.wait_reasons_json) == []


def test_engine_is_deterministic_and_reproducible():
    s = small_settings()
    _, frames = trending_frames()
    eng = DeterministicForecaster(s, events=EventCalendar([]))
    slot = MON_00 + 3900
    a = eng.forecast(frames, slot, CTX, created_at=from_epoch_min(slot))[0].forecast.with_hash()
    b = DeterministicForecaster(s, events=EventCalendar([])).forecast(frames, slot, CTX, created_at=from_epoch_min(slot))[0].forecast.with_hash()
    assert a.forecast_id == b.forecast_id and a.content_hash == b.content_hash


def test_no_lookahead_future_candles_do_not_change_forecast():
    s = small_settings()
    df, full = trending_frames()
    slot = MON_00 + 3600
    past = MarketFrames(df[df.ts_min < slot], s.features.timeframes)
    eng = DeterministicForecaster(s, events=EventCalendar([]))
    a = eng.forecast(past, slot, CTX, created_at=from_epoch_min(slot))[0].forecast
    b = eng.forecast(full, slot, CTX, created_at=from_epoch_min(slot))[0].forecast
    assert a.with_hash().content_hash == b.with_hash().content_hash


def test_late_creation_starts_evaluation_after_issue_time():
    s = small_settings()
    _, frames = trending_frames()
    eng = DeterministicForecaster(s, events=EventCalendar([]))
    slot = MON_00 + 3600
    f = eng.forecast(frames, slot, CTX, created_at=from_epoch_min(slot) + timedelta(seconds=20))[0].forecast
    assert f.start_min == slot + 1  # never credits price moves before the forecast existed
    assert f.as_of_ts < f.start_ts


# ----------------------------------------------------------------------------- WAIT conditions
def _forecast(s, frames, slot, events=None):
    eng = DeterministicForecaster(s, events=events or EventCalendar([]))
    return eng.forecast(frames, slot, CTX, created_at=from_epoch_min(slot))[0].forecast


def test_wait_score_below_threshold_keeps_candidate():
    s = small_settings(forecasting={"thresholds": {"up": 0.99, "down": -0.99}})
    _, frames = trending_frames()
    f = _forecast(s, frames, MON_00 + 3900)
    assert f.direction == "WAIT" and "score_below_threshold" in f.wait_reasons
    assert f.candidate_direction == "UP" and f.target_price is not None  # WAIT candidates are still evaluated
    assert f.confidence == "none"


def test_wait_stale_data_has_no_candidate():
    s = small_settings()
    df, _ = trending_frames()
    slot = MON_00 + 3900
    frames = MarketFrames(df[df.ts_min < slot - 10], s.features.timeframes)  # last 10 minutes missing
    f = _forecast(s, frames, slot)
    assert f.direction == "WAIT" and f.primary_wait_reason == "data_stale"
    assert f.candidate_direction == "NONE" and f.target_price is None


def test_wait_insufficient_data():
    s = small_settings()
    df, _ = trending_frames(n=600)
    frames = MarketFrames(df, s.features.timeframes)
    f = _forecast(s, frames, MON_00 + 600)
    assert f.direction == "WAIT" and "insufficient_data" in f.wait_reasons and f.composite_score is None


def test_wait_data_quality_suspect_candle():
    s = small_settings()
    df, _ = trending_frames()
    df.loc[3890, "quality"] = "suspect"
    f = _forecast(s, MarketFrames(df, s.features.timeframes), MON_00 + 3900)
    assert "data_quality" in f.wait_reasons and f.candidate_direction == "NONE"


def test_wait_excessive_spread_and_quiet_market():
    s = small_settings()
    _, frames = trending_frames(spread=1.5)  # 2*spread floor dominates the distance
    f = _forecast(s, frames, MON_00 + 3900)
    assert "excessive_spread" in f.wait_reasons
    assert "volatility_invalid" in f.wait_reasons  # ATR/spread below the minimum


def test_wait_event_restriction():
    s = small_settings()
    _, frames = trending_frames()
    slot = MON_00 + 3900
    ev = EventCalendar([EconomicEvent("CPI", slot + 45, 30, 30)])
    f = _forecast(s, frames, slot, events=ev)
    assert "event_restriction" in f.wait_reasons


def test_wait_conflicting_signals(settings):
    s = small_settings(forecasting={"wait": {"max_conflict": 0.0, "conflict_min_magnitude": 0.0}})
    _, frames = trending_frames(drift=0.0, seed=9)
    f = _forecast(s, frames, MON_00 + 3900)
    assert "conflicting_signals" in f.wait_reasons


def test_wait_session_close_and_market_closed():
    # Real calendar (with daily break). Friday 2026-01-09 21:30 UTC: 60m horizon crosses the weekly close.
    s = load_settings(overrides={"features": small_settings().features.model_dump()}, include_local=False)
    fri = to_epoch_min("2026-01-09T21:30:00Z")
    rng = np.random.default_rng(4)
    closes = 2000 + np.cumsum(0.3 * rng.standard_normal(4000))
    df = make_candles(closes, start_min=fri - 4000)
    cal = MarketCalendar(s.market)
    df = df[cal.open_mask(df.ts_min.to_numpy())]
    frames = MarketFrames(df, s.features.timeframes)
    f = _forecast(s, frames, fri)
    assert "session_close" in f.wait_reasons
    saturday = fri + 24 * 60
    f2 = _forecast(s, frames, saturday)
    assert f2.primary_wait_reason == "market_closed"
