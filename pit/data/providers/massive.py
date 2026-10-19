"""Massive (formerly Polygon.io) REST adapter.

* Candles: ``GET /v2/aggs/ticker/C:XAUUSD/range/1/minute/{from_ms}/{to_ms}`` (paged via
  ``next_url``). Aggregates carry one price series; spreads are filled from the configured
  default and labelled ``assumed``. Confirm the aggregate price convention in Massive's
  documentation before relying on it for execution analysis.
* Latest quote: ``GET /v1/last_quote/currencies/XAU/USD`` (bid/ask).

``data_mode`` defaults to ``delayed`` because lower-tier entitlements are delayed; set it to
``live`` only on a confirmed real-time plan.
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd

from pit.common.timeutils import ensure_utc, to_epoch_min
from pit.data.models import ProviderCapabilities, Quote, conform_candle_frame
from pit.data.providers.base import MarketDataProvider, ProviderError
from pit.data.providers.http import HttpJsonClient


def to_massive_ticker(instrument: str) -> str:
    return "C:" + instrument.replace("/", "").upper()


class MassiveProvider(MarketDataProvider):
    name = "massive"
    quote_convention = "provider_aggregate"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.massive.com",
        requests_per_minute: int = 5,
        data_mode: str = "delayed",
        instruments: tuple[str, ...] = ("XAU/USD", "AUD/USD"),
        transport: httpx.BaseTransport | None = None,
        ops=None,
        sleep=None,
        max_pages: int = 50,
    ):
        self.api_key = api_key
        self.data_mode = data_mode
        self.instruments = instruments
        self.max_pages = max_pages
        kwargs = {"sleep": sleep} if sleep else {}
        self.http = HttpJsonClient(self.name, base_url, requests_per_minute, transport=transport, ops=ops, **kwargs)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            instruments=self.instruments,
            historical_candles=True,
            live_quotes=True,
            bid_ask_candles=False,
            tick_quotes=True,
            streaming=True,
            requires_credentials=True,
            quote_convention=self.quote_convention,
            notes="Aggregates without bid/ask; last-quote endpoint has bid/ask. Real-time requires a higher tier.",
        )

    def fetch_candles(self, instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
        start, end = ensure_utc(start), ensure_utc(end)
        s_ms, e_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000) - 1
        path = f"/v2/aggs/ticker/{to_massive_ticker(instrument)}/range/1/minute/{s_ms}/{e_ms}"
        params: dict = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}
        frames = []
        for _ in range(self.max_pages):
            payload = self.http.get_json(path, params)
            frames.append(self._parse(payload))
            next_url = payload.get("next_url") if isinstance(payload, dict) else None
            if not next_url:
                break
            parsed = urlparse(next_url)
            path = parsed.path
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            params["apiKey"] = self.api_key
        df = pd.concat(frames, ignore_index=True) if frames else conform_candle_frame(
            pd.DataFrame({"ts_min": pd.Series(dtype="int64")})
        )
        s, e = to_epoch_min(start), to_epoch_min(end)
        return df[(df["ts_min"] >= s) & (df["ts_min"] < e)].reset_index(drop=True)

    def _parse(self, payload: dict) -> pd.DataFrame:
        if not isinstance(payload, dict):
            raise ProviderError("massive: unexpected response type")
        status = str(payload.get("status", "")).upper()
        if status in ("ERROR", "NOT_AUTHORIZED"):
            raise ProviderError(f"massive: {payload.get('error') or payload.get('message') or status}")
        results = payload.get("results") or []
        if not results:
            return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
        raw = pd.DataFrame(results)
        out = pd.DataFrame(
            {
                "ts_min": (raw["t"].astype("int64") // 60000).astype("int64"),
                "open": raw["o"].astype(float),
                "high": raw["h"].astype(float),
                "low": raw["l"].astype(float),
                "close": raw["c"].astype(float),
                "volume": raw.get("v", pd.Series([None] * len(raw))).astype(float),
            }
        )
        out["quote_convention"] = self.quote_convention
        out["data_mode"] = self.data_mode
        return conform_candle_frame(out)

    def fetch_latest_quote(self, instrument: str) -> Quote | None:
        base, quote = instrument.upper().split("/")
        payload = self.http.get_json(f"/v1/last_quote/currencies/{base}/{quote}", {"apiKey": self.api_key})
        try:
            last = payload["last"]
            ts = pd.Timestamp(int(last["timestamp"]), unit="ms", tz="UTC").to_pydatetime()
            return Quote(
                instrument=instrument,
                source=self.name,
                ts=ts,
                bid=float(last["bid"]),
                ask=float(last["ask"]),
                data_mode=self.data_mode,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"massive: malformed last-quote payload: {exc}") from exc

    def close(self) -> None:
        self.http.close()
