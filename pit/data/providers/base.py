"""Provider-independent market-data interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from pit.data.models import ProviderCapabilities, Quote


class ProviderError(Exception):
    """Base class for provider failures."""

    retryable: bool = False

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ProviderConfigurationError(ProviderError):
    """Missing credentials or invalid provider configuration."""


class AuthenticationError(ProviderError):
    """Credentials rejected (401/403). Never retried."""


class RateLimitError(ProviderError):
    retryable = True


class ProviderUnavailableError(ProviderError):
    """Transient failure (network, 5xx). Retried with backoff."""

    retryable = True


class InstrumentNotSupportedError(ProviderError):
    pass


class MarketDataProvider(ABC):
    """A source of 1-minute candles (and optionally quotes) for one or more instruments.

    Implementations must:
      * return only COMPLETED candles with ``ts_min`` in ``[start, end)``;
      * never fill, interpolate or otherwise fabricate missing prices;
      * state their quote convention and data mode honestly.
    """

    name: str = "abstract"
    data_mode: str = "historical"
    quote_convention: str = "unknown"

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abstractmethod
    def fetch_candles(self, instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
        """1-minute candles in ``[start, end)`` as a candle frame (see :mod:`pit.data.models`)."""

    def fetch_latest_quote(self, instrument: str) -> Quote | None:
        """Latest bid/ask quote, or ``None`` if the provider cannot supply one."""
        return None

    def supports(self, instrument: str) -> bool:
        return instrument in self.capabilities.instruments

    def close(self) -> None:  # pragma: no cover - default no-op
        pass
