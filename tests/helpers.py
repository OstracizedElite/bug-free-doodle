"""Deterministic test data builders with known expected results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pit.common.timeutils import to_epoch_min
from pit.data.models import conform_candle_frame

# Monday 2026-01-05 10:00 UTC: market open (05:00 New York, winter time).
MONDAY_10_UTC = to_epoch_min("2026-01-05T10:00:00Z")


def make_candles(
    closes,
    start_min: int = MONDAY_10_UTC,
    spread: float = 0.2,
    wick: float = 0.1,
    opens=None,
    highs=None,
    lows=None,
    step: int = 1,
) -> pd.DataFrame:
    """Build a 1-minute candle frame from closes. Open = previous close unless given."""
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    if opens is None:
        opens = np.concatenate([[closes[0]], closes[:-1]])
    opens = np.asarray(opens, dtype=float)
    if highs is None:
        highs = np.maximum(opens, closes) + wick
    if lows is None:
        lows = np.minimum(opens, closes) - wick
    highs, lows = np.asarray(highs, dtype=float), np.asarray(lows, dtype=float)
    half = spread / 2.0
    df = pd.DataFrame(
        {
            "ts_min": start_min + np.arange(n, dtype="int64") * step,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "bid_open": opens - half,
            "bid_high": highs - half,
            "bid_low": lows - half,
            "bid_close": closes - half,
            "ask_open": opens + half,
            "ask_high": highs + half,
            "ask_low": lows + half,
            "ask_close": closes + half,
            "spread": spread,
            "spread_source": "observed",
            "volume": 10.0,
            "quote_convention": "test_mid",
            "data_mode": "historical",
            "quality": "ok",
            "quality_notes": None,
        }
    )
    return conform_candle_frame(df)


def random_walk_closes(n: int, start: float = 2000.0, sigma: float = 0.5, drift: float = 0.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    return start + np.cumsum(drift + sigma * rng.standard_normal(n))


def make_forecast(
    direction: str = "UP",
    reference: float = 2000.0,
    distance: float = 1.0,
    start_min: int = MONDAY_10_UTC,
    horizon: int = 60,
    stop_multiple: float = 1.0,
    candidate: str | None = None,
    confidence: str = "medium",
    session: str = "london",
    forecast_id: str | None = None,
    p_hit: float | None = 0.6,
    model_type: str = "deterministic",
):
    """Minimal immutable Forecast plus its primary/mirror targets."""
    from pit.common.timeutils import iso_min
    from pit.forecasting.models import Forecast, ForecastTarget

    cand = candidate or (direction if direction in ("UP", "DOWN") else "NONE")
    sgn = 1 if cand == "UP" else -1
    fid = forecast_id or f"fc_test_{start_min}_{cand}_{direction}"
    has_target = cand in ("UP", "DOWN")
    f = Forecast(
        forecast_id=fid, origin="backtest", run_id="t", data_mode="historical", instrument="XAU/USD",
        provider="test", quote_convention="test_mid", model_type=model_type, model_version="test",
        config_version="cfg", slot_ts=iso_min(start_min), slot_min=start_min, as_of_ts=iso_min(start_min),
        created_at=iso_min(start_min), start_ts=iso_min(start_min), start_min=start_min,
        expiry_ts=iso_min(start_min + horizon), expiry_min=start_min + horizon, horizon_minutes=horizon,
        reference_price=reference, reference_price_type="mid_close_1m", bid=reference - 0.1, ask=reference + 0.1,
        spread=0.2, spread_source="observed", direction=direction, candidate_direction=cand,
        composite_score=0.5 * sgn if has_target else 0.0, components_json="{}",
        target_price=round(reference + sgn * distance, 2) if has_target else None,
        target_distance=distance if has_target else None, target_atr_multiple=0.75 if has_target else None,
        stop_price=round(reference - sgn * stop_multiple * distance, 2) if has_target else None,
        atr_1m=distance / 0.75, sigma_1m=0.0003, p_hit=p_hit if has_target else None, p_hit_method="uncalibrated_heuristic",
        p_hit_formula="test", confidence=confidence if direction in ("UP", "DOWN") else "none", regime="uncertain",
        vol_regime="normal", trend_regime="none", session=session,
    )
    targets = []
    if has_target:
        opp = "DOWN" if cand == "UP" else "UP"
        targets = [
            ForecastTarget(forecast_id=fid, label="primary", direction=cand, atr_multiple=0.75, distance=distance,
                           price=round(reference + sgn * distance, 2), is_primary=True),
            ForecastTarget(forecast_id=fid, label="mirror", direction=opp, atr_multiple=0.75, distance=distance,
                           price=round(reference - sgn * distance, 2), is_mirror=True),
        ]
    return f, targets
