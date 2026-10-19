"""Deterministic feature engine.

``FeatureEngine.compute(frames, as_of_min)`` uses only bars that completed at or before
``as_of_min`` and a fixed number of the most recent of them per timeframe. All formulas
are documented in ``docs/methodology.md`` (section "Features").
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pit.config.settings import FeaturesConfig
from pit.data.market_calendar import MarketCalendar
from pit.features import indicators as ind
from pit.features.frames import BarSeries, MarketFrames


def _r(x: Any, nd: int = 6) -> Any:
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), nd)
    return x


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


@dataclass
class FeatureSnapshot:
    instrument: str
    as_of_min: int
    reference_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    spread_source: str | None = None
    last_1m_end: int | None = None
    data_age_minutes: int | None = None
    sigma_1m: float | None = None
    atr_1m: float | None = None
    suspect_recent: int = 0
    timeframes: dict[str, dict[str, Any]] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, Any] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    input_hash: str = ""

    def tf(self, name: str) -> dict[str, Any]:
        return self.timeframes.get(name) or {"available": False}

    def payload(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "as_of_min": self.as_of_min,
            "reference_price": _r(self.reference_price, 5),
            "bid": _r(self.bid, 5),
            "ask": _r(self.ask, 5),
            "spread": _r(self.spread, 5),
            "spread_source": self.spread_source,
            "last_1m_end": self.last_1m_end,
            "data_age_minutes": self.data_age_minutes,
            "sigma_1m": _r(self.sigma_1m, 9),
            "atr_1m": _r(self.atr_1m, 6),
            "suspect_recent": self.suspect_recent,
            "missing_required": list(self.missing_required),
            "timeframes": {k: {kk: _r(vv) for kk, vv in v.items()} for k, v in self.timeframes.items()},
            "structure": {k: _r(v) for k, v in self.structure.items()},
            "regime": {k: _r(v) for k, v in self.regime.items()},
            "input_hash": self.input_hash,
        }

    @property
    def snapshot_id(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def features_available(self) -> bool:
        return not self.missing_required and self.reference_price is not None


class FeatureEngine:
    def __init__(self, cfg: FeaturesConfig, calendar: MarketCalendar, stale_after_minutes: float = 3.0,
                 suspect_lookback_minutes: int = 30):
        self.cfg = cfg
        self.calendar = calendar
        self.stale_after_minutes = stale_after_minutes
        self.suspect_lookback_minutes = suspect_lookback_minutes

    # ------------------------------------------------------------------ public
    def compute(self, frames: MarketFrames, as_of_min: int, instrument: str = "XAU/USD") -> FeatureSnapshot:
        cfg = self.cfg
        snap = FeatureSnapshot(instrument=instrument, as_of_min=int(as_of_min))
        windows: dict[str, BarSeries] = {tf: frames[tf].select(as_of_min, cfg.lookback_bars[tf]) for tf in cfg.timeframes}
        snap.input_hash = self._input_hash(windows)

        base = windows["1m"]
        if len(base) == 0:
            snap.missing_required = list(cfg.required_timeframes)
            return snap
        snap.reference_price = float(base.close[-1])
        snap.bid = float(base.bid_close[-1])
        snap.ask = float(base.ask_close[-1])
        snap.spread = float(base.spread[-1])
        snap.spread_source = frames.spread_source_at(as_of_min)
        snap.last_1m_end = int(base.end[-1])
        snap.data_age_minutes = int(as_of_min - base.end[-1])
        recent = base.start >= as_of_min - self.suspect_lookback_minutes
        snap.suspect_recent = int(np.sum(base.suspect & recent))

        r1 = ind.log_returns(base.close, base.start, max_step=1)[-cfg.vol_window:]
        if r1.size >= max(10, cfg.vol_window // 2):
            snap.sigma_1m = float(np.std(r1, ddof=1))
        snap.atr_1m = ind.atr(base.high, base.low, base.close, cfg.atr_period)

        for tf, bars in windows.items():
            snap.timeframes[tf] = self._timeframe_features(tf, bars, snap)

        snap.missing_required = [tf for tf in cfg.required_timeframes if not snap.tf(tf).get("available")]
        if snap.sigma_1m is None or not snap.sigma_1m > 0:
            snap.missing_required.append("sigma_1m")
        if snap.atr_1m is None or not snap.atr_1m > 0:
            snap.missing_required.append("atr_1m")

        snap.structure = self._structure(windows.get(cfg.structure_timeframe), snap)
        snap.regime = self._regime(snap)
        return snap

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _input_hash(windows: dict[str, BarSeries]) -> str:
        h = hashlib.sha256()
        for tf in sorted(windows):
            b = windows[tf]
            h.update(tf.encode())
            for arr in (b.start, b.open, b.high, b.low, b.close, b.spread):
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def _timeframe_features(self, tf: str, b: BarSeries, snap: FeatureSnapshot) -> dict[str, Any]:
        cfg = self.cfg
        n = len(b)
        out: dict[str, Any] = {"available": False, "n_bars": n}
        if n < cfg.min_bars or snap.reference_price is None:
            return out
        p = snap.reference_price
        fast_p, mid_p, slow_p = cfg.ema_periods
        atr_v = ind.atr(b.high, b.low, b.close, cfg.atr_period)
        ema_f = ind.ema_series(b.close, fast_p)
        ema_m = ind.ema_series(b.close, mid_p)
        ema_s = ind.ema_series(b.close, slow_p)
        if atr_v is None or atr_v <= 0 or ema_f is None or ema_m is None or ema_s is None:
            return out
        baseline = ind.mean_true_range(b.high, b.low, b.close, cfg.vol_baseline_bars)
        vol_ratio = atr_v / baseline if baseline else None
        lag = cfg.slope_lag
        slope = (ema_m[-1] - ema_m[-1 - lag]) / (lag * atr_v) if ema_m.size > lag else None
        nb = cfg.consistency_bars
        rets = ind.log_returns(b.close)[-nb:]

        # Momentum: reference price vs the close of the last bar ending at/before T - window.
        window = cfg.momentum_lookback_bars * b.minutes
        idx = int(np.searchsorted(b.end, snap.as_of_min - window, side="right")) - 1
        mom_change = mom_ret = mom_z = None
        w_eff = None
        if idx >= 0 and snap.sigma_1m and snap.last_1m_end is not None:
            p_hist = float(b.close[idx])
            w_eff = self.calendar.count_open_minutes(int(b.end[idx]), int(snap.last_1m_end))
            mom_change = p - p_hist
            mom_ret = (p / p_hist - 1.0) * 100.0
            if w_eff > 0:
                mom_z = math.log(p / p_hist) / (snap.sigma_1m * math.sqrt(w_eff))

        hi = float(b.high[-nb:].max())
        lo = float(b.low[-nb:].min())
        hc = self.cfg.regime
        vol_state = None
        if vol_ratio is not None:
            vol_state = "expanding" if vol_ratio >= hc.high_vol_ratio else ("contracting" if vol_ratio <= hc.low_vol_ratio else "normal")
        out.update(
            {
                "available": True,
                "last_bar_end": int(b.end[-1]),
                "close": float(b.close[-1]),
                "atr": atr_v,
                "tr_baseline": baseline,
                "vol_ratio": vol_ratio,
                "vol_state": vol_state,
                "ema_fast": float(ema_f[-1]),
                "ema_mid": float(ema_m[-1]),
                "ema_slow": float(ema_s[-1]),
                "ema_mid_slope_atr": slope,
                "ema_sep_atr": (float(ema_f[-1]) - float(ema_m[-1])) / atr_v,
                "sma": ind.sma(b.close, cfg.sma_period),
                "price_vs_ema_mid_atr": (p - float(ema_m[-1])) / atr_v,
                "momentum_change": mom_change,
                "momentum_return_pct": mom_ret,
                "momentum_z": mom_z,
                "momentum_window_min": w_eff,
                "consistency": ind.directional_consistency(b.close, nb),
                "up_fraction": ind.up_fraction(b.close, nb),
                "efficiency": ind.efficiency_ratio(b.close, nb),
                "range_high": hi,
                "range_low": lo,
                "range_atr": (hi - lo) / atr_v,
                "return_std": float(np.std(rets, ddof=1)) if rets.size > 1 else None,
                "realised_vol": float(np.sqrt(np.sum(rets**2))) if rets.size else None,
            }
        )
        return out

    def _structure(self, b: BarSeries | None, snap: FeatureSnapshot) -> dict[str, Any]:
        cfg = self.cfg
        out: dict[str, Any] = {"available": False}
        if b is None or snap.reference_price is None:
            return out
        b = b.slice(max(0, len(b) - cfg.structure_lookback_bars), len(b))
        tf_feats = snap.tf(cfg.structure_timeframe)
        atr_v = tf_feats.get("atr")
        if len(b) < cfg.swing_left + cfg.swing_right + 3 or not atr_v:
            return out
        p = snap.reference_price
        sh, sl = ind.swing_points(b.high, b.low, cfg.swing_left, cfg.swing_right)
        highs, lows = b.high[sh], b.low[sl]

        pattern = 0.0
        if highs.size >= 2:
            pattern += 0.5 * _sign(highs[-1] - highs[-2])
        if lows.size >= 2:
            pattern += 0.5 * _sign(lows[-1] - lows[-2])

        # Breakout levels: the most recent confirmed swing high/low; if none exists in the window
        # (common in persistent trends) fall back to the prior range extreme, excluding the most
        # recent swing_right + 1 bars so the level is not defined by the move being tested.
        prior_n = len(b) - (cfg.swing_right + 1)
        if highs.size:
            res_level, res_idx, res_kind = float(highs[-1]), int(sh[-1]), "swing"
        else:
            res_idx = int(np.argmax(b.high[:prior_n]))
            res_level, res_kind = float(b.high[res_idx]), "range"
        if lows.size:
            sup_level, sup_idx, sup_kind = float(lows[-1]), int(sl[-1]), "swing"
        else:
            sup_idx = int(np.argmin(b.low[:prior_n]))
            sup_level, sup_kind = float(b.low[sup_idx]), "range"

        breakout = 0.0
        label = "none"
        if p > res_level:
            breakout, label = 1.0, "breakout_up"
        elif p < sup_level:
            breakout, label = -1.0, "breakout_down"
        elif res_idx + 1 < len(b) and b.high[res_idx + 1:].max() > res_level:
            # Failed breakout: a later bar traded through the level, but price is back inside.
            breakout, label = -0.5, "failed_breakout_up"
        elif sup_idx + 1 < len(b) and b.low[sup_idx + 1:].min() < sup_level:
            breakout, label = 0.5, "failed_breakout_down"

        above = highs[highs > p]
        below = lows[lows < p]
        resistance = float(above.min()) if above.size else None
        support = float(below.max()) if below.size else None
        rng_hi, rng_lo = float(b.high.max()), float(b.low.min())
        nb = cfg.consistency_bars
        recent_range_atr = (float(b.high[-nb:].max()) - float(b.low[-nb:].min())) / atr_v
        out.update(
            {
                "available": True,
                "timeframe": cfg.structure_timeframe,
                "swing_high_count": int(highs.size),
                "swing_low_count": int(lows.size),
                "last_swing_high": float(highs[-1]) if highs.size else None,
                "prev_swing_high": float(highs[-2]) if highs.size > 1 else None,
                "last_swing_low": float(lows[-1]) if lows.size else None,
                "prev_swing_low": float(lows[-2]) if lows.size > 1 else None,
                "swing_pattern": pattern,
                "breakout_level_high": res_level,
                "breakout_level_high_kind": res_kind,
                "breakout_level_low": sup_level,
                "breakout_level_low_kind": sup_kind,
                "breakout_signal": breakout,
                "breakout_label": label,
                "score": float(np.clip(0.5 * pattern + 0.5 * breakout, -1.0, 1.0)),
                "resistance": resistance,
                "support": support,
                "dist_resistance_atr": (resistance - p) / atr_v if resistance is not None else None,
                "dist_support_atr": (p - support) / atr_v if support is not None else None,
                "range_high": rng_hi,
                "range_low": rng_lo,
                "range_position": (p - rng_lo) / (rng_hi - rng_lo) if rng_hi > rng_lo else 0.5,
                "recent_range_atr": recent_range_atr,
                "consolidating": bool(recent_range_atr < cfg.regime.consolidation_range_atr),
            }
        )
        return out

    def _regime(self, snap: FeatureSnapshot) -> dict[str, Any]:
        rc = self.cfg.regime
        f = snap.tf(rc.timeframe)
        if not f.get("available") or snap.reference_price is None:
            return {"regime": "uncertain", "vol_regime": "unknown", "trend_regime": "unknown", "reason": "insufficient_data"}
        vol_ratio = f.get("vol_ratio")
        slope = f.get("ema_mid_slope_atr") or 0.0
        p = snap.reference_price
        ef, em, es = f["ema_fast"], f["ema_mid"], f["ema_slow"]
        if vol_ratio is None:
            vol_regime = "unknown"
        elif vol_ratio >= rc.high_vol_ratio:
            vol_regime = "high"
        elif vol_ratio <= rc.low_vol_ratio:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        consolidating = bool(snap.structure.get("consolidating"))
        if ef > em > es and slope >= rc.trend_slope_min and p > em:
            trend_regime = "up"
        elif ef < em < es and slope <= -rc.trend_slope_min and p < em:
            trend_regime = "down"
        elif abs(slope) < rc.trend_slope_min and consolidating:
            trend_regime = "range"
        else:
            trend_regime = "none"
        if vol_regime == "high":
            regime = "high_volatility"
        elif trend_regime == "up":
            regime = "uptrend"
        elif trend_regime == "down":
            regime = "downtrend"
        elif vol_regime == "low":
            regime = "low_volatility"
        elif trend_regime == "range":
            regime = "range_bound"
        else:
            regime = "uncertain"
        return {
            "regime": regime,
            "vol_regime": vol_regime,
            "trend_regime": trend_regime,
            "timeframe": rc.timeframe,
            "vol_ratio": vol_ratio,
            "ema_mid_slope_atr": slope,
            "consolidating": consolidating,
        }
