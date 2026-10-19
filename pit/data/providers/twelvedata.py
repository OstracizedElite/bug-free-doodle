"""Twelve Data REST adapter (https://twelvedata.com/docs).

* Candles: ``GET /time_series?symbol=XAU/USD&interval=1min`` (max 5000 rows per call).
* Latest price: ``GET /price``.

Twelve Data publishes a single price per bar (no bid/ask), so spreads are filled from the
configured default and labelled ``assumed``. Its prices are an aggregated reference, not
any broker's executable CFD price.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pandas as pd

from pit.common.timeutils import ensure_utc, to_epoch_min
from pit.data.models import ProviderCapabilities, conform_candle_frame
from pit.data.providers.base import MarketDataProvider, ProviderError
from pit.data.providers.http import HttpJsonClient
from pit.data.quality import timestamps_to_epoch_min

MAX_ROWS = 5000


class TwelveDataProvider(MarketDataProvider):
    name = "twelvedata"
    quote_convention = "provider_mid"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.twelvedata.com",
        requests_per_minute: int = 8,
        data_mode: str = "live",
        instruments: tuple[str, ...] = ("XAU/USD", "AUD/USD"),
        transport: httpx.BaseTransport | None = None,
        ops=None,
        sleep=None,
    ):
        self.api_key = api_key
        self.data_mode = data_mode
        self.instruments = instruments
        kwargs = {"sleep": sleep} if sleep else {}
        self.http = HttpJsonClient(
            self.name, base_url, requests_per_minute, transport=transport, ops=ops, **kwargs
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            instruments=self.instruments,
            historical_candles=True,
            live_quotes=False,
            bid_ask_candles=False,
            tick_quotes=False,
            streaming=True,  # WebSocket on paid plans (not implemented in V1)
            requires_credentials=True,
            quote_convention=self.quote_convention,
            notes="Single price per bar; spreads are assumed. WebSocket requires a paid plan.",
        )

    def fetch_candles(self, instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
        start, end = ensure_utc(start), ensure_utc(end)
        frames = []
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + timedelta(minutes=MAX_ROWS))
            payload = self.http.get_json(
                "/time_series",
                {
                    "symbol": instrument,
                    "interval": "1min",
                    "start_date": cursor.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_date": chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "timezone": "UTC",
                    "order": "ASC",
                    "outputsize": MAX_ROWS,
                    "apikey": self.api_key,
                },
            )
            frames.append(self._parse(payload))
            cursor = chunk_end
        if not frames:
            return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
        df = pd.concat(frames, ignore_index=True)
        s, e = to_epoch_min(start), to_epoch_min(end)
        return df[(df["ts_min"] >= s) & (df["ts_min"] < e)].reset_index(drop=True)

    def _parse(self, payload: dict) -> pd.DataFrame:
        if not isinstance(payload, dict):
            raise ProviderError("twelvedata: unexpected response type")
        if payload.get("status") == "error":
            code = payload.get("code")
            message = payload.get("message", "unknown error")
            # 'No data is available on the specified dates' is an empty result, not a failure.
            if code == 400 and "no data" in message.lower():
                return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
            raise ProviderError(f"twelvedata error {code}: {message}", code)
        values = payload.get("values") or []
        if not values:
            return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
        raw = pd.DataFrame(values)
        minutes, aligned = timestamps_to_epoch_min(raw["datetime"], timezone="UTC")
        raw = raw[aligned].copy()
        raw["ts_min"] = minutes[aligned]
        for c in ("open", "high", "low", "close", "volume"):
            if c in raw.columns:
                raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw["quote_convention"] = self.quote_convention
        raw["data_mode"] = self.data_mode
        return conform_candle_frame(raw.drop(columns=["datetime"]))

    def fetch_latest_price(self, instrument: str) -> float | None:
        payload = self.http.get_json("/price", {"symbol": instrument, "apikey": self.api_key})
        try:
            return float(payload["price"])
        except (KeyError, TypeError, ValueError):
            return None

    def close(self) -> None:
        self.http.close()
