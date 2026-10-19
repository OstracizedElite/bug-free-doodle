"""Construct the configured market-data provider."""

from __future__ import annotations

import os

from pit.config.settings import Settings
from pit.data.market_calendar import MarketCalendar
from pit.data.providers.base import MarketDataProvider, ProviderConfigurationError

PROVIDER_NAMES = ("synthetic", "csv", "twelvedata", "oanda", "massive")


def _require_env(var: str, provider: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise ProviderConfigurationError(
            f"provider {provider!r} needs the environment variable {var} (see .env.example). "
            "Paid or account-based providers must be approved before use; "
            "set data.provider: synthetic to continue without credentials."
        )
    return value


def create_provider(settings: Settings, ops=None, name: str | None = None) -> MarketDataProvider:
    name = name or settings.data.provider
    p = settings.providers
    instruments = (settings.instrument.symbol, settings.instrument.fx_symbol)
    if name == "synthetic":
        from pit.data.providers.synthetic import SyntheticProvider

        return SyntheticProvider(
            p.synthetic,
            MarketCalendar(settings.market),
            instrument=settings.instrument.symbol,
            fx_symbol=settings.instrument.fx_symbol,
        )
    if name == "csv":
        from pit.data.providers.csv_file import CsvProvider

        if not p.csv.path:
            raise ProviderConfigurationError("providers.csv.path is empty")
        return CsvProvider(
            settings.resolve_path(p.csv.path),
            instrument=settings.instrument.symbol,
            quote_convention=p.csv.quote_convention,
            data_mode=p.csv.data_mode,
            source_name=p.csv.source_name,
        )
    if name == "twelvedata":
        from pit.data.providers.twelvedata import TwelveDataProvider

        return TwelveDataProvider(
            _require_env(p.twelvedata.api_key_env, name),
            base_url=p.twelvedata.base_url,
            requests_per_minute=p.twelvedata.requests_per_minute,
            data_mode=p.twelvedata.data_mode,
            instruments=instruments,
            ops=ops,
        )
    if name == "oanda":
        from pit.data.providers.oanda import OandaProvider

        return OandaProvider(
            _require_env(p.oanda.api_key_env, name),
            account_id=os.environ.get(p.oanda.account_id_env) or None,
            environment=p.oanda.environment,
            requests_per_minute=p.oanda.requests_per_minute,
            data_mode=p.oanda.data_mode,
            instruments=instruments,
            ops=ops,
        )
    if name == "massive":
        from pit.data.providers.massive import MassiveProvider

        return MassiveProvider(
            _require_env(p.massive.api_key_env, name),
            base_url=p.massive.base_url,
            requests_per_minute=p.massive.requests_per_minute,
            data_mode=p.massive.data_mode,
            instruments=instruments,
            ops=ops,
        )
    raise ProviderConfigurationError(f"unknown provider {name!r}; expected one of {PROVIDER_NAMES}")
