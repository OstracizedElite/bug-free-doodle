"""Volatility-adjusted target selection.

Primary distance  D = max(atr_multiple * ATR14(1m), spread_multiple * spread)   (USD/oz)
UP target         reference + D           DOWN target   reference - D
Alternatives      k * ATR14(1m) for k in alternative_atr_multiples (no spread floor)
Mirror target     the primary distance in the OPPOSITE direction - a built-in control:
                  a directional edge requires the target to be hit more often than its mirror.
Stop (hypothetical, for executable evaluation) = reference -/+ stop_multiple * D.

The reference price is the mid close of the last completed 1-minute candle. Target touches
measured on mid prices are NOT executable outcomes; executable results are evaluated
separately on bid/ask (see pit.evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass

from pit.config.settings import TargetsConfig


@dataclass(frozen=True)
class TargetLevel:
    label: str
    direction: str  # UP | DOWN
    atr_multiple: float | None
    distance: float
    price: float
    is_primary: bool = False
    is_mirror: bool = False


def target_distance(atr_1m: float, spread: float, cfg: TargetsConfig) -> float:
    return max(cfg.atr_multiple * atr_1m, cfg.spread_multiple * spread)


def build_targets(direction: str, reference: float, atr_1m: float, spread: float, cfg: TargetsConfig,
                  decimals: int = 2) -> list[TargetLevel]:
    if direction not in ("UP", "DOWN"):
        raise ValueError("targets require a candidate direction")
    sgn = 1.0 if direction == "UP" else -1.0
    opposite = "DOWN" if direction == "UP" else "UP"
    d = target_distance(atr_1m, spread, cfg)
    levels = [
        TargetLevel("primary", direction, cfg.atr_multiple, d, round(reference + sgn * d, decimals), is_primary=True),
        TargetLevel("mirror", opposite, cfg.atr_multiple, d, round(reference - sgn * d, decimals), is_mirror=True),
    ]
    for k in cfg.alternative_atr_multiples:
        dk = k * atr_1m
        levels.append(TargetLevel(f"atr_{k:g}", direction, k, dk, round(reference + sgn * dk, decimals)))
    return levels


def stop_price(direction: str, reference: float, distance: float, stop_multiple: float, decimals: int = 2) -> float:
    sgn = 1.0 if direction == "UP" else -1.0
    return round(reference - sgn * stop_multiple * distance, decimals)
