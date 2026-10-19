"""Heuristic target-hit probability (P-HIT).

THIS IS AN UNCALIBRATED HEURISTIC, NOT A STATISTICALLY CALIBRATED PROBABILITY. Its
calibration is measured (Brier score, reliability tables) but never auto-tuned in V1.

Version ``phit-heuristic-v1``:

  sigma_P = sigma_1m * reference                      (USD per sqrt(minute))
  p0      = 2 * (1 - Phi(D / (sigma_P * sqrt(H))))    driftless Brownian first-passage
                                                      probability of moving D within H minutes
  A       = direction_sign * ma_alignment             trend alignment in [-1, 1]
  logit(p) = logit(clip(p0, 0.01, 0.99))
           + score_coef     * (|S| - score_pivot)
           + alignment_coef * A
           - spread_coef    * (spread / D)
  P-HIT   = clip(sigmoid(logit(p)), min, max)
"""

from __future__ import annotations

import math

from scipy.stats import norm

from pit.config.settings import ProbabilityConfig

METHOD_LABEL = "uncalibrated_heuristic"


def formula_text(cfg: ProbabilityConfig) -> str:
    return (
        f"{cfg.version}: p0=2*(1-Phi(D/(sigma_1m*ref*sqrt(H)))); "
        f"logit(p)=logit(clip(p0,0.01,0.99))+{cfg.score_coef}*(|S|-{cfg.score_pivot})"
        f"+{cfg.alignment_coef}*A-{cfg.spread_coef}*(spread/D); p=clip(sigmoid,{cfg.min},{cfg.max})"
    )


def barrier_touch_probability(distance: float, sigma_price_per_sqrt_min: float, horizon_minutes: float) -> float:
    """P(max_{t<=H} W_t >= D) for driftless Brownian motion with volatility sigma."""
    if distance <= 0:
        return 1.0
    if sigma_price_per_sqrt_min <= 0 or horizon_minutes <= 0:
        return 0.0
    return float(2.0 * (1.0 - norm.cdf(distance / (sigma_price_per_sqrt_min * math.sqrt(horizon_minutes)))))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def heuristic_p_hit(
    score: float,
    direction_sign: int,
    ma_alignment: float,
    distance: float,
    spread: float,
    sigma_1m: float,
    reference: float,
    horizon_minutes: float,
    cfg: ProbabilityConfig,
) -> tuple[float, dict]:
    p0 = barrier_touch_probability(distance, sigma_1m * reference, horizon_minutes)
    p0c = min(0.99, max(0.01, p0))
    alignment = direction_sign * ma_alignment
    spread_ratio = spread / distance if distance > 0 else 1.0
    z = (
        _logit(p0c)
        + cfg.score_coef * (abs(score) - cfg.score_pivot)
        + cfg.alignment_coef * alignment
        - cfg.spread_coef * spread_ratio
    )
    p = 1.0 / (1.0 + math.exp(-z))
    p = min(cfg.max, max(cfg.min, p))
    return p, {"p0": p0, "alignment": alignment, "spread_ratio": spread_ratio, "logit": z}
