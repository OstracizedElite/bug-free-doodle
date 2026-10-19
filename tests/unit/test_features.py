import math

import numpy as np
import pandas as pd
import pytest

from pit.common.timeutils import to_epoch_min

from pit.config import load_settings
from pit.data.market_calendar import MarketCalendar
from pit.features.engine import FeatureEngine
from pit.features.frames import MarketFrames, resample
from tests.helpers import make_candles, random_walk_closes

# Monday 2026-01-05 00:00 UTC (Sunday 19:00 New York): market open.
MON_00 = to_epoch_min("2026-01-05T00:00:00Z")
NO_BREAK = {"market": {"daily_break_start": "17:00", "daily_break_end": "17:00"}}


def small_settings(**extra):
    o = {
        **NO_BREAK,
        "features": {
            "lookback_bars": {"1m": 240, "5m": 120, "15m": 120, "30m": 120, "60m": 120, "4h": 60, "12h": 30},
            "min_bars": 20,
            "required_timeframes": ["1m", "5m", "15m", "60m"],
        },
    }
    o.update(extra)
    return load_settings(overrides=o, include_local=False)


def frames_for(closes, start=MON_00, spread=0.2):
    df = make_candles(closes, start_min=start, spread=spread)
    return df, MarketFrames(df, ["1m", "5m", "15m", "30m", "60m", "4h", "12h"])


def test_resample_matches_pandas_and_excludes_incomplete_bars():
    closes = random_walk_closes(200)
    df, frames = frames_for(closes)
    idx = pd.to_datetime(df["ts_min"] * 60, unit="s", utc=True)
    ref = df.set_index(idx)[["open", "high", "low", "close"]].resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    b = frames["15m"]
    assert np.allclose(b.open, ref["open"]) and np.allclose(b.high, ref["high"])
    assert np.allclose(b.low, ref["low"]) and np.allclose(b.close, ref["close"])
    # as_of = 10 min into a 15m bar: that bar is incomplete and must be excluded.
    as_of = MON_00 + 15 * 5 + 10
    sel = b.select(as_of, 100)
    assert sel.end[-1] <= as_of
    assert sel.end[-1] == MON_00 + 75


def test_features_do_not_change_when_future_data_is_appended():
    s = small_settings()
    eng = FeatureEngine(s.features, MarketCalendar(s.market))
    closes = random_walk_closes(10000, seed=3)
    as_of = MON_00 + 9000
    _, past_only = frames_for(closes[:9000])
    _, with_future = frames_for(closes)
    a = eng.compute(past_only, as_of)
    b = eng.compute(with_future, as_of)
    assert a.payload() == b.payload()
    assert a.snapshot_id == b.snapshot_id
    # Fixed lookback windows: dropping history older than the windows changes nothing either.
    start_trim = 9000 - 125 * 60  # keep slightly more than the 60m lookback (120 bars)
    _, trimmed = frames_for(closes[start_trim:9000], start=MON_00 + start_trim)
    c = eng.compute(trimmed, as_of)
    for tf in ("1m", "5m", "15m", "60m"):
        assert c.tf(tf) == a.tf(tf)
    assert c.structure == a.structure


def test_momentum_z_exact():
    s = small_settings()
    eng = FeatureEngine(s.features, MarketCalendar(s.market))
    closes = random_walk_closes(2000, seed=11)
    _, frames = frames_for(closes)
    as_of = MON_00 + 1800
    snap = eng.compute(frames, as_of)
    p = closes[1799]
    r = np.diff(np.log(closes[:1800]))[-60:]
    sigma = np.std(r, ddof=1)
    assert snap.sigma_1m == pytest.approx(sigma)
    # 5m momentum: last 5m bar ending at/before as_of-5 ends at as_of-5 -> close index 1794.
    p_hist = closes[1794]
    expected_z = math.log(p / p_hist) / (sigma * math.sqrt(5))
    assert snap.tf("5m")["momentum_z"] == pytest.approx(expected_z)
    assert snap.tf("5m")["momentum_change"] == pytest.approx(p - p_hist)
    assert snap.tf("5m")["momentum_return_pct"] == pytest.approx((p / p_hist - 1) * 100)


def test_atr_cross_check_with_pandas_on_60m():
    s = small_settings()
    eng = FeatureEngine(s.features, MarketCalendar(s.market))
    closes = random_walk_closes(4000, seed=5)
    df, frames = frames_for(closes)
    as_of = MON_00 + 3600
    snap = eng.compute(frames, as_of)
    idx = pd.to_datetime(df["ts_min"] * 60, unit="s", utc=True)
    h = df.set_index(idx)[["high", "low", "close"]].resample("60min").agg({"high": "max", "low": "min", "close": "last"})
    h = h[h.index + pd.Timedelta(minutes=60) <= pd.Timestamp(as_of * 60, unit="s", tz="UTC")].tail(120)
    prev = h["close"].shift(1)
    tr = pd.concat([h["high"] - h["low"], (h["high"] - prev).abs(), (h["low"] - prev).abs()], axis=1).max(axis=1)
    assert snap.tf("60m")["atr"] == pytest.approx(tr.iloc[-14:].mean())


def test_insufficient_history_reported():
    s = small_settings()
    eng = FeatureEngine(s.features, MarketCalendar(s.market))
    _, frames = frames_for(random_walk_closes(300))
    snap = eng.compute(frames, MON_00 + 300)
    assert "60m" in snap.missing_required  # only 5 hourly bars exist
    assert not snap.features_available


def test_regime_uptrend_and_breakout_on_trending_data():
    s = small_settings()
    eng = FeatureEngine(s.features, MarketCalendar(s.market))
    rng = np.random.default_rng(2)
    closes = 2000 + np.cumsum(0.08 + 0.3 * rng.standard_normal(4000))
    _, frames = frames_for(closes)
    snap = eng.compute(frames, MON_00 + 4000)
    assert snap.regime["trend_regime"] == "up"
    assert snap.regime["regime"] in ("uptrend", "high_volatility")
    assert snap.structure["available"]
    assert snap.tf("60m")["ema_fast"] > snap.tf("60m")["ema_mid"] > snap.tf("60m")["ema_slow"]


def test_suspect_candles_counted_and_staleness():
    s = small_settings()
    eng = FeatureEngine(s.features, MarketCalendar(s.market), suspect_lookback_minutes=30)
    df = make_candles(random_walk_closes(3000), start_min=MON_00)
    df.loc[2990, "quality"] = "suspect"
    frames = MarketFrames(df, s.features.timeframes)
    snap = eng.compute(frames, MON_00 + 3000)
    assert snap.suspect_recent == 1 and snap.data_age_minutes == 0
    snap2 = eng.compute(frames, MON_00 + 3010)  # 10 minutes with no new candles
    assert snap2.data_age_minutes == 10
