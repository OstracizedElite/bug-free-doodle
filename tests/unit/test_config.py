import pytest
from pydantic import ValidationError

from pit.config import load_settings
from pit.config.settings import deep_merge


def test_defaults_load_and_match_specification(settings):
    f = settings.forecasting
    assert settings.instrument.symbol == "XAU/USD"
    assert f.interval_minutes == 15
    assert f.horizons == [60]
    assert f.thresholds.up == 0.35 and f.thresholds.down == -0.35
    w = f.weights
    assert (w.m1, w.m5, w.m15, w.trend60, w.ma_alignment, w.structure, w.vol_consistency) == (
        0.10, 0.20, 0.25, 0.20, 0.10, 0.10, 0.05,
    )
    assert f.targets.atr_multiple == 0.75
    assert f.targets.spread_multiple == 2.0
    assert f.targets.alternative_atr_multiples == [0.5, 1.0, 1.5, 2.0]
    assert settings.paper.initial_balance == 50000.0
    assert settings.paper.base_currency == "AUD"
    assert settings.llm.enabled is False
    assert settings.data.provider == "synthetic"
    assert set(settings.features.timeframes) == {"1m", "5m", "15m", "30m", "60m", "4h", "12h"}


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        load_settings(overrides={"forecasting": {"weights": {"m1": 0.5}}}, include_local=False)


def test_unsupported_horizon_rejected():
    with pytest.raises(ValidationError):
        load_settings(overrides={"forecasting": {"horizons": [45]}}, include_local=False)


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        load_settings(overrides={"forecasting": {"bogus": 1}}, include_local=False)


def test_config_version_is_stable_and_sensitive(make_settings):
    a = make_settings()
    b = make_settings()
    assert a.forecast_config_version == b.forecast_config_version
    c = make_settings({"forecasting": {"thresholds": {"up": 0.4}}})
    assert c.forecast_config_version != a.forecast_config_version
    # Paper settings do not change the forecast version, and vice versa.
    d = make_settings({"paper": {"risk_per_trade": 0.01}})
    assert d.forecast_config_version == a.forecast_config_version
    assert d.paper_config_version != a.paper_config_version


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    override = {"a": {"b": 5}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 5, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
