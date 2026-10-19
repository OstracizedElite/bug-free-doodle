"""Component normalisation and the composite directional score.

Each component is mapped deterministically into [-1, +1] (see docs/methodology.md):

  m1, m5, m15     tanh(z / momentum_scale), z = ln(P / P_hist) / (sigma_1m * sqrt(W))
  trend60         tanh(0.5 * sep / trend_sep_scale + 0.5 * slope / trend_slope_scale)
                  sep   = (EMA_fast - EMA_mid) / ATR on 60m bars
                  slope = (EMA_mid[t] - EMA_mid[t-lag]) / (lag * ATR) on 60m bars
  ma_alignment    mean over alignment timeframes of
                  [sign(P - EMA_fast) + sign(EMA_fast - EMA_mid) + sign(EMA_mid - EMA_slow)] / 3
  structure       clip(0.5 * swing_pattern + 0.5 * breakout_signal, -1, 1) on 15m bars
  vol_consistency signed efficiency ratio over the last N consistency-timeframe bars

Composite S = sum(weight_i * component_i), in [-1, +1] because weights sum to 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pit.config.settings import ForecastingConfig, FeaturesConfig
from pit.features.engine import FeatureSnapshot

COMPONENTS = ("m1", "m5", "m15", "trend60", "ma_alignment", "structure", "vol_consistency")


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def momentum_component(z: float | None, scale: float) -> float | None:
    if z is None:
        return None
    return math.tanh(z / scale)


def trend_component(sep_atr: float | None, slope_atr: float | None, sep_scale: float, slope_scale: float) -> float | None:
    if sep_atr is None or slope_atr is None:
        return None
    return math.tanh(0.5 * sep_atr / sep_scale + 0.5 * slope_atr / slope_scale)


def alignment_for_timeframe(price: float, ema_fast: float, ema_mid: float, ema_slow: float) -> float:
    return (_sign(price - ema_fast) + _sign(ema_fast - ema_mid) + _sign(ema_mid - ema_slow)) / 3.0


@dataclass(frozen=True)
class ScoreResult:
    components: dict[str, float | None]
    weights: dict[str, float]
    score: float | None
    conflict: float | None
    complete: bool

    def as_dict(self) -> dict:
        return {
            "components": {k: (None if v is None else round(v, 6)) for k, v in self.components.items()},
            "weights": self.weights,
            "score": None if self.score is None else round(self.score, 6),
            "conflict": None if self.conflict is None else round(self.conflict, 6),
            "complete": self.complete,
        }


def compute_components(snap: FeatureSnapshot, fc: ForecastingConfig, feat: FeaturesConfig) -> dict[str, float | None]:
    norm = fc.normalisation
    p = snap.reference_price
    comps: dict[str, float | None] = {}
    comps["m1"] = momentum_component(snap.tf("1m").get("momentum_z"), norm.momentum_scale)
    comps["m5"] = momentum_component(snap.tf("5m").get("momentum_z"), norm.momentum_scale)
    comps["m15"] = momentum_component(snap.tf("15m").get("momentum_z"), norm.momentum_scale)
    t60 = snap.tf("60m")
    comps["trend60"] = trend_component(
        t60.get("ema_sep_atr"), t60.get("ema_mid_slope_atr"), norm.trend_sep_scale, norm.trend_slope_scale
    )
    aligns = []
    for tf in feat.alignment_timeframes:
        f = snap.tf(tf)
        if f.get("available") and p is not None:
            aligns.append(alignment_for_timeframe(p, f["ema_fast"], f["ema_mid"], f["ema_slow"]))
    comps["ma_alignment"] = sum(aligns) / len(aligns) if len(aligns) == len(feat.alignment_timeframes) else None
    comps["structure"] = snap.structure.get("score") if snap.structure.get("available") else None
    comps["vol_consistency"] = snap.tf(norm.consistency_timeframe).get("efficiency")
    return comps


def composite_score(components: dict[str, float | None], fc: ForecastingConfig) -> ScoreResult:
    weights = fc.weights.model_dump()
    complete = all(components.get(k) is not None for k in COMPONENTS)
    if not complete:
        return ScoreResult(components, weights, None, None, False)
    s = sum(weights[k] * float(components[k]) for k in COMPONENTS)
    s = max(-1.0, min(1.0, s))
    min_mag = fc.wait.conflict_min_magnitude
    if s == 0:
        conflict = 0.5
    else:
        conflict = sum(
            weights[k] for k in COMPONENTS if abs(components[k]) >= min_mag and _sign(components[k]) != _sign(s)
        )
    return ScoreResult(components, weights, s, conflict, True)
