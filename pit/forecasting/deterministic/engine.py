"""PIT Version 1.0 deterministic forecaster.

For a forecast slot ``T`` (data cut-off: candles completed at or before ``T``):

1. compute features from completed bars only;
2. compute the seven normalised components and the composite score S;
3. evaluate every WAIT condition and record *all* that apply (first = primary);
4. derive the candidate direction (sign of S) and, where features exist, targets,
   a hypothetical stop and the heuristic P-HIT - also for WAITs, so the value of each
   WAIT filter can itself be measured;
5. emit one immutable record per configured horizon.

WAIT is a legitimate forecast outcome, never an error.
"""

from __future__ import annotations

import json
from datetime import datetime

from pit import MODEL_VERSION
from pit.common.timeutils import ceil_minute, iso, iso_min, to_epoch_min
from pit.config.settings import Settings
from pit.data.market_calendar import MarketCalendar
from pit.features.engine import FeatureEngine, FeatureSnapshot
from pit.features.frames import MarketFrames
from pit.forecasting.deterministic import probability as prob
from pit.forecasting.deterministic.scoring import ScoreResult, composite_score, compute_components
from pit.forecasting.deterministic.targets import build_targets, stop_price, target_distance
from pit.forecasting.events import EventCalendar
from pit.forecasting.models import (
    INTEGRITY_REASONS,
    Direction,
    Forecast,
    ForecastContext,
    ForecastResult,
    ForecastTarget,
    WaitReason,
    forecast_id_for,
)

MODEL_TYPE = "deterministic"


def classify_confidence(direction: str, score: float | None, p_hit: float | None, settings: Settings) -> str:
    if direction == Direction.WAIT.value or score is None:
        return "none"
    c = settings.forecasting.confidence
    s = abs(score)
    if s >= c.high_min_score and (p_hit or 0.0) >= c.high_min_phit:
        return "high"
    if s >= c.medium_min_score:
        return "medium"
    return "low"


class DeterministicForecaster:
    def __init__(self, settings: Settings, calendar: MarketCalendar | None = None, events: EventCalendar | None = None):
        self.settings = settings
        self.calendar = calendar or MarketCalendar(settings.market)
        if events is None:
            events = EventCalendar.from_file(settings.resolve_path(settings.events.file), settings.events)
        self.events = events
        self.feature_engine = FeatureEngine(
            settings.features,
            self.calendar,
            stale_after_minutes=settings.data.stale_after_seconds / 60.0,
            suspect_lookback_minutes=settings.data.sanity.suspect_lookback_minutes,
        )
        self.config_version = settings.forecast_config_version

    # ------------------------------------------------------------------ public
    def compute_features(self, frames: MarketFrames, as_of_min: int) -> FeatureSnapshot:
        return self.feature_engine.compute(frames, as_of_min, self.settings.instrument.symbol)

    def forecast(
        self,
        frames: MarketFrames,
        slot_min: int,
        ctx: ForecastContext,
        created_at: datetime,
        horizons: list[int] | None = None,
        snapshot: FeatureSnapshot | None = None,
    ) -> list[ForecastResult]:
        snap = snapshot or self.compute_features(frames, slot_min)
        comps = compute_components(snap, self.settings.forecasting, self.settings.features) if snap.reference_price else {}
        score = composite_score(comps, self.settings.forecasting) if comps else ScoreResult({}, {}, None, None, False)
        out = []
        for h in horizons or self.settings.forecasting.horizons:
            out.append(self._build(snap, score, slot_min, h, ctx, created_at))
        return out

    # ------------------------------------------------------------------ internals
    def _wait_reasons(self, snap: FeatureSnapshot, score: ScoreResult, slot_min: int, start_min: int,
                      expiry_min: int, distance: float | None) -> list[str]:
        s = self.settings
        w = s.forecasting.wait
        reasons: list[str] = []
        if not self.calendar.is_valid_forecast_slot(slot_min):
            reasons.append(WaitReason.MARKET_CLOSED.value)
        stale_minutes = s.data.stale_after_seconds / 60.0
        if snap.data_age_minutes is None or snap.data_age_minutes > stale_minutes:
            reasons.append(WaitReason.DATA_STALE.value)
        if snap.suspect_recent > 0:
            reasons.append(WaitReason.DATA_QUALITY.value)
        if not snap.features_available or not score.complete:
            reasons.append(WaitReason.INSUFFICIENT_DATA.value)
        if self.events.restricting(start_min, expiry_min):
            reasons.append(WaitReason.EVENT_RESTRICTION.value)
        if w.block_session_close and self.calendar.first_closed_minute(start_min, expiry_min) is not None:
            reasons.append(WaitReason.SESSION_CLOSE.value)
        if distance and snap.spread is not None and snap.spread / distance > w.max_spread_to_target:
            reasons.append(WaitReason.EXCESSIVE_SPREAD.value)
        vol_ratio = snap.regime.get("vol_ratio")
        too_volatile = vol_ratio is not None and vol_ratio > w.max_vol_ratio
        too_quiet = bool(snap.atr_1m and snap.spread and snap.atr_1m / snap.spread < w.min_atr_to_spread)
        if too_volatile or too_quiet:
            reasons.append(WaitReason.VOLATILITY_INVALID.value)
        if score.conflict is not None and score.conflict > w.max_conflict:
            reasons.append(WaitReason.CONFLICTING_SIGNALS.value)
        if score.score is not None:
            th = s.forecasting.thresholds
            if th.down < score.score < th.up:
                reasons.append(WaitReason.SCORE_BELOW_THRESHOLD.value)
        return reasons

    def _build(self, snap: FeatureSnapshot, score: ScoreResult, slot_min: int, horizon: int,
               ctx: ForecastContext, created_at: datetime) -> ForecastResult:
        s = self.settings
        start_min = max(slot_min, to_epoch_min(ceil_minute(created_at)))
        expiry_min = start_min + horizon
        fid = forecast_id_for(ctx.origin, ctx.run_id, ctx.instrument, slot_min, horizon, MODEL_TYPE, self.config_version)

        candidate = "NONE"
        if score.score is not None and score.score != 0:
            candidate = Direction.UP.value if score.score > 0 else Direction.DOWN.value

        distance = None
        if snap.atr_1m and snap.spread is not None:
            distance = target_distance(snap.atr_1m, snap.spread, s.forecasting.targets)

        reasons = self._wait_reasons(snap, score, slot_min, start_min, expiry_min, distance)
        direction = candidate if not reasons else Direction.WAIT.value
        if candidate == "NONE":
            direction = Direction.WAIT.value
            if not reasons:  # exact zero score
                reasons = [WaitReason.SCORE_BELOW_THRESHOLD.value]

        targets: list[ForecastTarget] = []
        target_price = stop = p_hit = None
        p_details: dict = {}
        decimals = s.instrument.price_decimals
        # A candidate is only meaningful on trustworthy, complete data.
        integrity_ok = snap.features_available and score.complete and not (set(reasons) & INTEGRITY_REASONS)
        if not integrity_ok:
            candidate = "NONE"
        if candidate in ("UP", "DOWN") and distance and snap.reference_price is not None:
            levels = build_targets(candidate, snap.reference_price, snap.atr_1m, snap.spread, s.forecasting.targets, decimals)
            targets = [ForecastTarget(forecast_id=fid, **lvl.__dict__) for lvl in levels]
            target_price = levels[0].price
            stop = stop_price(candidate, snap.reference_price, distance, s.evaluation.stop_multiple, decimals)
            p_hit, p_details = prob.heuristic_p_hit(
                score=score.score,
                direction_sign=1 if candidate == "UP" else -1,
                ma_alignment=score.components.get("ma_alignment") or 0.0,
                distance=distance,
                spread=snap.spread,
                sigma_1m=snap.sigma_1m,
                reference=snap.reference_price,
                horizon_minutes=horizon,
                cfg=s.forecasting.probability,
            )

        components = score.as_dict()
        components.update(
            {
                "p_hit_details": {k: round(v, 6) for k, v in p_details.items()},
                "structure_label": snap.structure.get("breakout_label"),
                "data_age_minutes": snap.data_age_minutes,
                "suspect_recent": snap.suspect_recent,
                "missing_required": snap.missing_required,
                "thresholds": s.forecasting.thresholds.model_dump(),
            }
        )
        record = Forecast(
            forecast_id=fid,
            origin=ctx.origin,
            run_id=ctx.run_id,
            data_mode=ctx.data_mode,
            instrument=ctx.instrument,
            provider=ctx.provider,
            quote_convention=ctx.quote_convention,
            model_type=MODEL_TYPE,
            model_version=MODEL_VERSION,
            config_version=self.config_version,
            slot_ts=iso_min(slot_min),
            slot_min=slot_min,
            as_of_ts=iso_min(slot_min),
            created_at=iso(created_at),
            start_ts=iso_min(start_min),
            start_min=start_min,
            expiry_ts=iso_min(expiry_min),
            expiry_min=expiry_min,
            horizon_minutes=horizon,
            reference_price=snap.reference_price,
            reference_price_type=s.forecasting.targets.reference,
            bid=snap.bid,
            ask=snap.ask,
            spread=snap.spread,
            spread_source=snap.spread_source,
            direction=direction,
            candidate_direction=candidate,
            composite_score=None if score.score is None else round(score.score, 6),
            components_json=json.dumps(components, sort_keys=True),
            target_price=target_price,
            target_distance=None if distance is None or not targets else round(distance, 6),
            target_atr_multiple=s.forecasting.targets.atr_multiple if targets else None,
            stop_price=stop,
            atr_1m=snap.atr_1m,
            sigma_1m=snap.sigma_1m,
            p_hit=None if p_hit is None else round(p_hit, 6),
            p_hit_method=prob.METHOD_LABEL if p_hit is not None else None,
            p_hit_formula=prob.formula_text(s.forecasting.probability) if p_hit is not None else None,
            confidence=classify_confidence(direction, score.score, p_hit, s),
            regime=snap.regime.get("regime"),
            vol_regime=snap.regime.get("vol_regime"),
            trend_regime=snap.regime.get("trend_regime"),
            session=self.calendar.session_label(slot_min),
            wait_reasons_json=json.dumps(reasons),
            primary_wait_reason=reasons[0] if direction == Direction.WAIT.value and reasons else None,
            snapshot_id=snap.snapshot_id,
        )
        return ForecastResult(forecast=record, targets=targets, snapshot=snap)
