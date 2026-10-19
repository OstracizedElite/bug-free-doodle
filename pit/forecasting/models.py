"""Forecast records. Once stored, a forecast is immutable (enforced by database triggers)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from pit.features.engine import FeatureSnapshot


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    WAIT = "WAIT"


class WaitReason(str, Enum):
    MARKET_CLOSED = "market_closed"
    DATA_STALE = "data_stale"
    DATA_QUALITY = "data_quality"
    INSUFFICIENT_DATA = "insufficient_data"
    EVENT_RESTRICTION = "event_restriction"
    SESSION_CLOSE = "session_close"
    EXCESSIVE_SPREAD = "excessive_spread"
    VOLATILITY_INVALID = "volatility_invalid"
    CONFLICTING_SIGNALS = "conflicting_signals"
    SCORE_BELOW_THRESHOLD = "score_below_threshold"
    LLM_REJECTED = "llm_rejected"
    DETERMINISTIC_WAIT = "deterministic_wait"


# Data-integrity reasons: no candidate direction is derived and an LLM may never override them.
INTEGRITY_REASONS = {
    WaitReason.MARKET_CLOSED.value,
    WaitReason.DATA_STALE.value,
    WaitReason.DATA_QUALITY.value,
    WaitReason.INSUFFICIENT_DATA.value,
}


class ForecastTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    forecast_id: str
    label: str
    direction: str
    atr_multiple: float | None
    distance: float
    price: float
    is_primary: bool = False
    is_mirror: bool = False


class Forecast(BaseModel):
    model_config = ConfigDict(frozen=True)

    forecast_id: str
    origin: str  # live | backtest
    run_id: str = ""
    data_mode: str
    instrument: str
    provider: str
    quote_convention: str
    model_type: str  # deterministic | llm_assisted
    parent_forecast_id: str | None = None
    model_version: str
    config_version: str
    slot_ts: str
    slot_min: int
    as_of_ts: str
    created_at: str
    start_ts: str
    start_min: int
    expiry_ts: str
    expiry_min: int
    horizon_minutes: int
    reference_price: float | None
    reference_price_type: str
    bid: float | None
    ask: float | None
    spread: float | None
    spread_source: str | None
    direction: str
    candidate_direction: str
    composite_score: float | None
    components_json: str | None
    target_price: float | None
    target_distance: float | None
    target_atr_multiple: float | None
    stop_price: float | None
    atr_1m: float | None
    sigma_1m: float | None
    p_hit: float | None
    p_hit_method: str | None
    p_hit_formula: str | None
    confidence: str
    regime: str | None
    vol_regime: str | None
    trend_regime: str | None
    session: str | None
    wait_reasons_json: str = "[]"
    primary_wait_reason: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_prompt_version: str | None = None
    llm_call_id: str | None = None
    snapshot_id: str | None = None
    content_hash: str = ""

    @property
    def wait_reasons(self) -> list[str]:
        return json.loads(self.wait_reasons_json)

    @property
    def components(self) -> dict[str, Any]:
        return json.loads(self.components_json) if self.components_json else {}

    def canonical_payload(self) -> dict[str, Any]:
        d = self.model_dump()
        d.pop("content_hash", None)
        return d

    def computed_hash(self) -> str:
        blob = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def with_hash(self) -> "Forecast":
        return self.model_copy(update={"content_hash": self.computed_hash()})


def forecast_id_for(origin: str, run_id: str, instrument: str, slot_min: int, horizon: int, model_type: str,
                    config_version: str) -> str:
    key = f"{origin}|{run_id}|{instrument}|{slot_min}|{horizon}|{model_type}|{config_version}"
    return "fc_" + hashlib.sha256(key.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ForecastContext:
    origin: str
    data_mode: str
    provider: str
    quote_convention: str
    instrument: str = "XAU/USD"
    run_id: str = ""


@dataclass
class ForecastResult:
    forecast: Forecast
    targets: list[ForecastTarget] = field(default_factory=list)
    snapshot: FeatureSnapshot | None = None
