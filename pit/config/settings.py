"""Typed configuration for PIT.

Configuration is layered: ``config/default.yaml`` -> ``config/local.yaml`` (optional,
git-ignored) -> the file named by ``$PIT_CONFIG`` (optional) -> programmatic overrides.

Two hashes are derived from the merged configuration:

* ``forecast_config_version`` - covers every parameter that can change a forecast. It is
  stored on every forecast record, so a parameter change always yields a distinct,
  separately evaluated configuration.
* ``paper_config_version`` - covers paper-trading parameters.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "local.yaml"

TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "4h": 240,
    "12h": 720,
}

SUPPORTED_HORIZONS = (15, 30, 60, 120)

DataModeName = Literal["synthetic", "historical", "delayed", "live"]
ProviderName = Literal["synthetic", "csv", "twelvedata", "oanda", "massive"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------- instrument
class InstrumentConfig(_Model):
    symbol: str = "XAU/USD"
    fx_symbol: str = "AUD/USD"
    price_decimals: int = 2


# --------------------------------------------------------------------------- data
class SanityConfig(_Model):
    min_price: float = 100.0
    max_price: float = 50000.0
    max_abs_1m_return: float = 0.03
    max_spread_fraction: float = 0.01
    suspect_lookback_minutes: int = 30


class DataConfig(_Model):
    provider: ProviderName = "synthetic"
    history_days: int = Field(100, ge=1)
    stale_after_seconds: int = Field(180, ge=30)
    poll_interval_seconds: int = Field(60, ge=5)
    backfill_chunk_minutes: int = Field(5000, ge=100)
    default_spread: float = Field(0.30, gt=0)
    sanity: SanityConfig = SanityConfig()


class SyntheticProviderConfig(_Model):
    seed: int = 20240101
    anchor: str = "2025-06-01T00:00:00Z"
    start_price: float = 3300.0
    annual_vol: float = 0.16
    vol_of_vol: float = 0.35
    vol_persistence_minutes: float = 720.0
    drift_persistence_minutes: float = 240.0
    drift_strength: float = 0.25
    base_spread: float = 0.25
    missing_rate: float = Field(0.0, ge=0.0, lt=1.0)
    fx_start_rate: float = 0.655


class CsvProviderConfig(_Model):
    path: str = ""
    quote_convention: str = "provider_mid"
    data_mode: DataModeName = "historical"
    source_name: str = "csv"


class TwelveDataProviderConfig(_Model):
    base_url: str = "https://api.twelvedata.com"
    api_key_env: str = "TWELVEDATA_API_KEY"
    requests_per_minute: int = 8
    data_mode: DataModeName = "live"


class OandaProviderConfig(_Model):
    environment: Literal["practice", "live"] = "practice"
    api_key_env: str = "OANDA_API_TOKEN"
    account_id_env: str = "OANDA_ACCOUNT_ID"
    requests_per_minute: int = 120
    data_mode: DataModeName = "live"


class MassiveProviderConfig(_Model):
    base_url: str = "https://api.massive.com"
    api_key_env: str = "MASSIVE_API_KEY"
    requests_per_minute: int = 5
    data_mode: DataModeName = "delayed"


class ProvidersConfig(_Model):
    synthetic: SyntheticProviderConfig = SyntheticProviderConfig()
    csv: CsvProviderConfig = CsvProviderConfig()
    twelvedata: TwelveDataProviderConfig = TwelveDataProviderConfig()
    oanda: OandaProviderConfig = OandaProviderConfig()
    massive: MassiveProviderConfig = MassiveProviderConfig()


# --------------------------------------------------------------------------- market
class MarketConfig(_Model):
    timezone: str = "America/New_York"
    week_open: str = "Sun 18:00"
    week_close: str = "Fri 17:00"
    daily_break_start: str = "17:00"
    daily_break_end: str = "18:00"
    holidays: list[str] = Field(default_factory=list)
    sessions_utc: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {
            "asia": (22, 7),
            "london": (7, 12),
            "overlap": (12, 16),
            "new_york": (16, 21),
            "late": (21, 22),
        }
    )


# --------------------------------------------------------------------------- features
class RegimeConfig(_Model):
    timeframe: str = "15m"
    high_vol_ratio: float = 1.5
    low_vol_ratio: float = 0.67
    trend_slope_min: float = 0.05
    consolidation_range_atr: float = 6.0


class FeaturesConfig(_Model):
    timeframes: list[str] = Field(default_factory=lambda: list(TIMEFRAME_MINUTES))
    required_timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "60m"])
    lookback_bars: dict[str, int] = Field(
        default_factory=lambda: {
            "1m": 240,
            "5m": 200,
            "15m": 200,
            "30m": 200,
            "60m": 200,
            "4h": 120,
            "12h": 100,
        }
    )
    min_bars: int = 60
    atr_period: int = 14
    vol_window: int = 60
    vol_baseline_bars: int = 96
    ema_periods: list[int] = Field(default_factory=lambda: [9, 21, 50])
    sma_period: int = 20
    slope_lag: int = 3
    consistency_bars: int = 12
    momentum_lookback_bars: int = 1
    swing_left: int = 2
    swing_right: int = 2
    structure_timeframe: str = "15m"
    structure_lookback_bars: int = 96
    alignment_timeframes: list[str] = Field(default_factory=lambda: ["5m", "15m", "60m"])
    regime: RegimeConfig = RegimeConfig()

    @model_validator(mode="after")
    def _check(self) -> "FeaturesConfig":
        for tf in self.timeframes + self.required_timeframes + self.alignment_timeframes:
            if tf not in TIMEFRAME_MINUTES:
                raise ValueError(f"unsupported timeframe {tf!r}")
        for tf in self.required_timeframes:
            if tf not in self.timeframes:
                raise ValueError(f"required timeframe {tf!r} missing from timeframes")
        for tf in self.timeframes:
            if tf not in self.lookback_bars:
                raise ValueError(f"lookback_bars missing for timeframe {tf!r}")
        if len(self.ema_periods) != 3 or sorted(self.ema_periods) != self.ema_periods:
            raise ValueError("ema_periods must be three ascending periods (fast, mid, slow)")
        if self.min_bars < self.atr_period + 1:
            raise ValueError("min_bars must exceed atr_period")
        return self


# --------------------------------------------------------------------------- forecasting
class ThresholdsConfig(_Model):
    up: float = 0.35
    down: float = -0.35

    @model_validator(mode="after")
    def _check(self) -> "ThresholdsConfig":
        if not (0 < self.up <= 1 and -1 <= self.down < 0):
            raise ValueError("thresholds must satisfy 0 < up <= 1 and -1 <= down < 0")
        return self


class WeightsConfig(_Model):
    m1: float = 0.10
    m5: float = 0.20
    m15: float = 0.25
    trend60: float = 0.20
    ma_alignment: float = 0.10
    structure: float = 0.10
    vol_consistency: float = 0.05

    @model_validator(mode="after")
    def _check(self) -> "WeightsConfig":
        values = self.model_dump()
        if any(v < 0 for v in values.values()):
            raise ValueError("weights must be non-negative")
        if abs(sum(values.values()) - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0 (got {sum(values.values()):.6f})")
        return self


class NormalisationConfig(_Model):
    momentum_scale: float = Field(2.0, gt=0)
    trend_sep_scale: float = Field(1.0, gt=0)
    trend_slope_scale: float = Field(0.2, gt=0)
    consistency_timeframe: str = "5m"


class TargetsConfig(_Model):
    reference: Literal["mid_close_1m"] = "mid_close_1m"
    atr_multiple: float = Field(0.75, gt=0)
    spread_multiple: float = Field(2.0, ge=0)
    alternative_atr_multiples: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])


class WaitConfig(_Model):
    max_spread_to_target: float = 0.35
    max_vol_ratio: float = 3.0
    min_atr_to_spread: float = 1.5
    max_conflict: float = 0.40
    conflict_min_magnitude: float = 0.10
    block_session_close: bool = True


class ProbabilityConfig(_Model):
    version: str = "phit-heuristic-v1"
    score_coef: float = 2.0
    score_pivot: float = 0.35
    alignment_coef: float = 0.25
    spread_coef: float = 1.0
    min: float = 0.02
    max: float = 0.98


class ConfidenceConfig(_Model):
    high_min_score: float = 0.60
    high_min_phit: float = 0.60
    medium_min_score: float = 0.45


class ForecastingConfig(_Model):
    interval_minutes: int = Field(15, ge=1)
    horizons: list[int] = Field(default_factory=lambda: [60])
    thresholds: ThresholdsConfig = ThresholdsConfig()
    weights: WeightsConfig = WeightsConfig()
    normalisation: NormalisationConfig = NormalisationConfig()
    targets: TargetsConfig = TargetsConfig()
    wait: WaitConfig = WaitConfig()
    probability: ProbabilityConfig = ProbabilityConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()

    @field_validator("horizons")
    @classmethod
    def _horizons(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("at least one horizon is required")
        for h in v:
            if h not in SUPPORTED_HORIZONS:
                raise ValueError(f"horizon {h} not in supported horizons {SUPPORTED_HORIZONS}")
        return sorted(set(v))

    @field_validator("interval_minutes")
    @classmethod
    def _interval(cls, v: int) -> int:
        if 1440 % v != 0:
            raise ValueError("interval_minutes must divide a day evenly")
        return v


class EventsConfig(_Model):
    file: str = "config/events.yaml"
    minutes_before: int = 30
    minutes_after: int = 30


# --------------------------------------------------------------------------- evaluation
class EvaluationConfig(_Model):
    version: str = "eval-v1"
    stop_multiple: float = Field(1.0, gt=0)
    slippage: float = Field(0.05, ge=0)
    min_coverage: float = Field(0.90, gt=0, le=1)
    ambiguity_policy: Literal["conservative"] = "conservative"


# --------------------------------------------------------------------------- paper trading
class FinancingConfig(_Model):
    long_annual_rate: float = 0.07
    short_annual_rate: float = 0.03
    rollover_time_ny: str = "17:00"


class PaperConfig(_Model):
    enabled: bool = True
    base_currency: Literal["AUD", "USD"] = "AUD"
    initial_balance: float = Field(50000.0, gt=0)
    leverage: float = Field(20.0, gt=0)
    margin_closeout_level: float = Field(0.50, gt=0, le=1)
    risk_per_trade: float = Field(0.005, gt=0, le=0.1)
    max_notional_multiple: float = Field(5.0, gt=0)
    stop_multiple: float = Field(1.0, gt=0)
    max_daily_loss: float = Field(0.02, gt=0, le=1)
    max_drawdown: float = Field(0.10, gt=0, le=1)
    max_concurrent_positions: int = Field(1, ge=1)
    allowed_sessions: list[str] = Field(
        default_factory=lambda: ["asia", "london", "overlap", "new_york"]
    )
    min_confidence: Literal["low", "medium", "high"] = "low"
    slippage: float = Field(0.05, ge=0)
    commission_per_unit: float = Field(0.0, ge=0)
    units_step: float = Field(1.0, gt=0)
    min_units: float = Field(1.0, gt=0)
    fx_max_age_minutes: int = 60
    financing: FinancingConfig = FinancingConfig()


# --------------------------------------------------------------------------- llm
class LLMPricing(_Model):
    input: float
    output: float


class LLMConfig(_Model):
    enabled: bool = False
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-5-mini"
    prompt_version: str = "llm-v1"
    only_directional: bool = True
    every_n_slots: int = Field(1, ge=1)
    monthly_budget_usd: float = Field(20.0, ge=0)
    max_output_tokens: int = 900
    timeout_seconds: float = 45
    allow_in_backtest: bool = False
    cache_responses: bool = True
    pricing_usd_per_mtok: dict[str, LLMPricing] = Field(default_factory=dict)


# --------------------------------------------------------------------------- ops
class SchedulerConfig(_Model):
    forecast_delay_seconds: int = 5
    grace_seconds: int = 90
    ingest_interval_seconds: int = 60
    evaluation_interval_seconds: int = 300
    paper_interval_seconds: int = 60
    heartbeat_interval_seconds: int = 60
    missed_slot_lookback_hours: int = 72


class CostsConfig(_Model):
    monthly_budget_usd: float = 150.0
    data_subscription_usd_per_month: float = 0.0


class DatabaseConfig(_Model):
    path: str = "var/pit.db"


class LoggingConfig(_Model):
    level: str = "INFO"
    file: str = "var/logs/pit.log"


# --------------------------------------------------------------------------- root
class Settings(_Model):
    instrument: InstrumentConfig = InstrumentConfig()
    data: DataConfig = DataConfig()
    providers: ProvidersConfig = ProvidersConfig()
    market: MarketConfig = MarketConfig()
    features: FeaturesConfig = FeaturesConfig()
    forecasting: ForecastingConfig = ForecastingConfig()
    events: EventsConfig = EventsConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    paper: PaperConfig = PaperConfig()
    llm: LLMConfig = LLMConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    costs: CostsConfig = CostsConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()

    # ------------------------------------------------------------------ hashing
    def forecast_config_dict(self) -> dict[str, Any]:
        """Every parameter that can change a deterministic forecast."""
        return {
            "instrument": self.instrument.model_dump(mode="json"),
            "data_sanity": self.data.sanity.model_dump(mode="json"),
            "stale_after_seconds": self.data.stale_after_seconds,
            "market": self.market.model_dump(mode="json"),
            "features": self.features.model_dump(mode="json"),
            "forecasting": self.forecasting.model_dump(mode="json"),
            "events": self.events.model_dump(mode="json"),
        }

    @property
    def forecast_config_version(self) -> str:
        return stable_hash(self.forecast_config_dict())[:12]

    def paper_config_dict(self) -> dict[str, Any]:
        return {"paper": self.paper.model_dump(mode="json"), "instrument": self.instrument.model_dump(mode="json")}

    @property
    def paper_config_version(self) -> str:
        return stable_hash(self.paper_config_dict())[:12]

    def resolve_path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def database_path(self) -> Path:
        env = os.environ.get("PIT_DATABASE_PATH")
        return self.resolve_path(env) if env else self.resolve_path(self.database.path)


def stable_hash(obj: Any) -> str:
    """SHA-256 of a canonical JSON encoding (sorted keys, no whitespace)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file {path} must contain a mapping")
    return data


def load_settings(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    include_local: bool = True,
) -> Settings:
    """Load layered configuration and validate it."""
    merged = _read_yaml(DEFAULT_CONFIG_PATH)
    if include_local:
        merged = deep_merge(merged, _read_yaml(LOCAL_CONFIG_PATH))
    extra = path or os.environ.get("PIT_CONFIG")
    if extra:
        extra_path = Path(extra)
        if not extra_path.is_absolute():
            extra_path = PROJECT_ROOT / extra_path
        if extra_path.resolve() != LOCAL_CONFIG_PATH.resolve() or not include_local:
            merged = deep_merge(merged, _read_yaml(extra_path))
    if overrides:
        merged = deep_merge(merged, overrides)
    return Settings.model_validate(merged)


def save_local_overrides(overrides: dict[str, Any]) -> Path:
    """Persist dashboard edits to config/local.yaml (merged with existing overrides)."""
    current = _read_yaml(LOCAL_CONFIG_PATH)
    merged = deep_merge(current, overrides)
    # Validate before writing so a bad edit can never break startup.
    Settings.model_validate(deep_merge(_read_yaml(DEFAULT_CONFIG_PATH), merged))
    LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(merged, fh, sort_keys=False)
    return LOCAL_CONFIG_PATH
