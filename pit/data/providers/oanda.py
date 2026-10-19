"""OANDA v20 REST adapter — READ-ONLY market data (https://developer.oanda.com/rest-live-v20/).

Only the instrument-candles and pricing endpoints are used. No order, trade or position
endpoints exist in this module, and none may be added without explicit authorisation.

* Candles: ``GET /v3/instruments/XAU_USD/candles?price=MBA&granularity=M1`` returns mid,
  bid and ask OHLC, so spreads are *observed*. Incomplete candles are discarded.
* Latest quote: ``GET /v3/accounts/{account}/pricing?instruments=XAU_USD``.

OANDA prices are that broker's CFD quotes (``quote_convention='broker_cfd_mid'``) and will
differ from other brokers (e.g. Plus500) and from spot reference prices.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pandas as pd

from pit.common.timeutils import ensure_utc, parse_ts, to_epoch_min
from pit.data.models import ProviderCapabilities, Quote, conform_candle_frame
from pit.data.providers.base import MarketDataProvider, ProviderError
from pit.data.providers.http import HttpJsonClient

MAX_CANDLES = 5000
HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}
STREAM_HOSTS = {
    "practice": "https://stream-fxpractice.oanda.com",
    "live": "https://stream-fxtrade.oanda.com",
}


def to_oanda_symbol(instrument: str) -> str:
    return instrument.replace("/", "_").upper()


def _parse_oanda_time(value: str) -> datetime:
    # OANDA returns RFC3339 with nanoseconds, e.g. 2026-10-19T14:00:00.000000000Z
    if "." in value:
        head, frac = value.rstrip("Z").split(".", 1)
        value = f"{head}.{frac[:6]}Z"
    return parse_ts(value)


class OandaProvider(MarketDataProvider):
    name = "oanda"
    quote_convention = "broker_cfd_mid"

    def __init__(
        self,
        api_token: str,
        account_id: str | None = None,
        environment: str = "practice",
        requests_per_minute: int = 120,
        data_mode: str = "live",
        instruments: tuple[str, ...] = ("XAU/USD", "AUD/USD"),
        transport: httpx.BaseTransport | None = None,
        ops=None,
        sleep=None,
    ):
        if environment not in HOSTS:
            raise ValueError(f"unknown OANDA environment {environment!r}")
        self.account_id = account_id
        self.environment = environment
        self.data_mode = data_mode
        self.instruments = instruments
        kwargs = {"sleep": sleep} if sleep else {}
        self.http = HttpJsonClient(
            self.name,
            HOSTS[environment],
            requests_per_minute,
            headers={"Authorization": f"Bearer {api_token}", "Accept-Datetime-Format": "RFC3339"},
            transport=transport,
            ops=ops,
            **kwargs,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            instruments=self.instruments,
            historical_candles=True,
            live_quotes=self.account_id is not None,
            bid_ask_candles=True,
            tick_quotes=True,
            streaming=True,
            requires_credentials=True,
            quote_convention=self.quote_convention,
            notes="Broker CFD bid/ask candles. Requires an OANDA v20 account and API token.",
        )

    def fetch_candles(self, instrument: str, start: datetime, end: datetime) -> pd.DataFrame:
        start, end = ensure_utc(start), ensure_utc(end)
        frames = []
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + timedelta(minutes=MAX_CANDLES - 1))
            payload = self.http.get_json(
                f"/v3/instruments/{to_oanda_symbol(instrument)}/candles",
                {
                    "price": "MBA",
                    "granularity": "M1",
                    "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )
            frames.append(self._parse(payload))
            cursor = chunk_end
        if not frames:
            return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
        df = pd.concat(frames, ignore_index=True).drop_duplicates("ts_min")
        s, e = to_epoch_min(start), to_epoch_min(end)
        return df[(df["ts_min"] >= s) & (df["ts_min"] < e)].reset_index(drop=True)

    def _parse(self, payload: dict) -> pd.DataFrame:
        if not isinstance(payload, dict) or "candles" not in payload:
            raise ProviderError(f"oanda: unexpected candle payload: {str(payload)[:200]}")
        rows = []
        for c in payload["candles"]:
            if not c.get("complete", False):
                continue  # never store an in-progress candle
            try:
                row = {"ts_min": to_epoch_min(_parse_oanda_time(c["time"])), "volume": float(c.get("volume", 0))}
                for side, prefix in (("mid", ""), ("bid", "bid_"), ("ask", "ask_")):
                    block = c.get(side)
                    if block:
                        row[f"{prefix}open"] = float(block["o"])
                        row[f"{prefix}high"] = float(block["h"])
                        row[f"{prefix}low"] = float(block["l"])
                        row[f"{prefix}close"] = float(block["c"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError(f"oanda: malformed candle {c}: {exc}") from exc
            rows.append(row)
        if not rows:
            return conform_candle_frame(pd.DataFrame({"ts_min": pd.Series(dtype="int64")}))
        df = pd.DataFrame(rows)
        df["quote_convention"] = self.quote_convention
        df["data_mode"] = self.data_mode
        return conform_candle_frame(df)

    def fetch_latest_quote(self, instrument: str) -> Quote | None:
        if not self.account_id:
            return None
        payload = self.http.get_json(
            f"/v3/accounts/{self.account_id}/pricing", {"instruments": to_oanda_symbol(instrument)}
        )
        try:
            p = payload["prices"][0]
            return Quote(
                instrument=instrument,
                source=self.name,
                ts=_parse_oanda_time(p["time"]),
                bid=float(p["bids"][0]["price"]),
                ask=float(p["asks"][0]["price"]),
                data_mode=self.data_mode,
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"oanda: malformed pricing payload: {exc}") from exc

    def close(self) -> None:
        self.http.close()
