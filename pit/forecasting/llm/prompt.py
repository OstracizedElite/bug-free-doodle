"""Versioned prompt for the LLM reviewer.

The model receives a compact, structured snapshot of derived market information - never
unbounded raw price history - and may only accept or reject the deterministic signal.
"""

from __future__ import annotations

import json
from typing import Any

from pit.common.timeutils import iso_min
from pit.features.engine import FeatureSnapshot
from pit.forecasting.events import EventCalendar
from pit.forecasting.models import Forecast

PROMPT_VERSION = "llm-v1"

SYSTEM_PROMPT = """You are a cautious quantitative analyst reviewing a short-term XAU/USD (spot gold) forecast
produced by a deterministic rule-based model. You are part of a research system that measures
whether your review improves forecast quality. No real money is at stake.

Rules:
1. Use ONLY the data in the user message. Do not invent, recall or estimate prices, levels,
   news or economic events. If information is missing, say so.
2. The economic events listed (if any) are the only verified events. Do not mention others.
3. You cannot change data-integrity decisions or the recorded forecast. You may only
   recommend ACCEPT or REJECT for the deterministic signal.
4. REJECT when the evidence is contradictory, the regime makes the signal unreliable, the
   target is small relative to costs, or risk is elevated. ACCEPT only with a clear rationale.
5. Respond with JSON matching the provided schema exactly. Do not include any price you
   were not given."""


def _r(x: Any, nd: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    return x


def build_input(forecast: Forecast, snapshot: FeatureSnapshot | None, events: EventCalendar | None) -> dict[str, Any]:
    comps = forecast.components
    data: dict[str, Any] = {
        "instrument": forecast.instrument,
        "forecast_time_utc": forecast.slot_ts,
        "horizon_minutes": forecast.horizon_minutes,
        "session": forecast.session,
        "price": {
            "reference_mid": forecast.reference_price,
            "bid": forecast.bid,
            "ask": forecast.ask,
            "spread_usd": _r(forecast.spread),
            "spread_source": forecast.spread_source,
            "quote_convention": forecast.quote_convention,
        },
        "deterministic_signal": {
            "direction": forecast.direction,
            "candidate_direction": forecast.candidate_direction,
            "composite_score": forecast.composite_score,
            "thresholds": comps.get("thresholds"),
            "components": comps.get("components"),
            "weights": comps.get("weights"),
            "conflict_weight": comps.get("conflict"),
            "wait_reasons": forecast.wait_reasons,
            "target_price": forecast.target_price,
            "target_distance_usd": _r(forecast.target_distance),
            "hypothetical_stop": forecast.stop_price,
            "heuristic_p_hit_uncalibrated": forecast.p_hit,
        },
        "regime": {"label": forecast.regime, "volatility": forecast.vol_regime, "trend": forecast.trend_regime},
    }
    if snapshot is not None:
        tfs = {}
        for tf in ("1m", "5m", "15m", "60m", "4h", "12h"):
            f = snapshot.tf(tf)
            if not f.get("available"):
                tfs[tf] = {"available": False}
                continue
            tfs[tf] = {
                "return_pct": _r(f.get("momentum_return_pct")),
                "momentum_z": _r(f.get("momentum_z"), 2),
                "atr_usd": _r(f.get("atr"), 3),
                "volatility_ratio": _r(f.get("vol_ratio"), 2),
                "ema_fast_minus_mid_atr": _r(f.get("ema_sep_atr"), 2),
                "ema_mid_slope_atr_per_bar": _r(f.get("ema_mid_slope_atr"), 3),
                "price_vs_ema_mid_atr": _r(f.get("price_vs_ema_mid_atr"), 2),
                "directional_consistency": _r(f.get("consistency"), 2),
                "efficiency_ratio": _r(f.get("efficiency"), 2),
            }
        s = snapshot.structure
        data["timeframes"] = tfs
        data["market_structure"] = {
            k: _r(s.get(k), 3)
            for k in ("breakout_label", "swing_pattern", "dist_resistance_atr", "dist_support_atr", "range_position",
                      "consolidating")
        }
        data["data_quality"] = {"data_age_minutes": snapshot.data_age_minutes, "suspect_candles_recent": snapshot.suspect_recent}
    if events is not None:
        upcoming = events.restricting(forecast.slot_min - 60, forecast.slot_min + 24 * 60)
        data["verified_economic_events"] = [{"name": e.name, "time_utc": iso_min(e.minute)} for e in upcoming]
    else:
        data["verified_economic_events"] = []
    return data


def render_user_message(payload: dict[str, Any]) -> str:
    return (
        "Review this deterministic XAU/USD forecast. Data (JSON):\n"
        + json.dumps(payload, sort_keys=True, default=str)
        + "\nReturn your assessment as JSON following the schema."
    )
