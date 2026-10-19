"""Tick-level bid/ask quote streaming with automatic reconnection.

Streams are optional: candles from REST polling are sufficient for forecasting. When a
streaming source is available, ticks are stored in the ``quotes`` table and used by the
evaluator to resolve intrabar ordering (stop vs target within the same minute).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable, Iterable, Iterator

import httpx

from pit.data.models import Quote
from pit.data.providers.http import redact
from pit.data.providers.oanda import STREAM_HOSTS, _parse_oanda_time, to_oanda_symbol

log = logging.getLogger(__name__)


class QuoteStream(ABC):
    source: str = "abstract"

    @abstractmethod
    def iter_quotes(self) -> Iterator[Quote]:
        """Yield quotes until the connection drops (then return or raise)."""


class OandaPriceStream(QuoteStream):
    source = "oanda"

    def __init__(
        self,
        api_token: str,
        account_id: str,
        instruments: Iterable[str],
        environment: str = "practice",
        data_mode: str = "live",
        transport: httpx.BaseTransport | None = None,
    ):
        self.instruments = list(instruments)
        self.data_mode = data_mode
        self.url = f"{STREAM_HOSTS[environment]}/v3/accounts/{account_id}/pricing/stream"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.transport = transport
        self._symbol_map = {to_oanda_symbol(i): i for i in self.instruments}

    def parse_line(self, line: str) -> Quote | None:
        if not line.strip():
            return None
        msg = json.loads(line)
        if msg.get("type") != "PRICE":
            return None  # HEARTBEAT and others
        try:
            return Quote(
                instrument=self._symbol_map.get(msg["instrument"], msg["instrument"].replace("_", "/")),
                source=self.source,
                ts=_parse_oanda_time(msg["time"]),
                bid=float(msg["bids"][0]["price"]),
                ask=float(msg["asks"][0]["price"]),
                data_mode=self.data_mode,
            )
        except (KeyError, IndexError, ValueError, TypeError):
            log.warning("oanda stream: malformed price message dropped")
            return None

    def iter_quotes(self) -> Iterator[Quote]:
        params = {"instruments": ",".join(to_oanda_symbol(i) for i in self.instruments)}
        with httpx.Client(transport=self.transport, timeout=httpx.Timeout(30.0, read=30.0)) as client:
            with client.stream("GET", self.url, params=params, headers=self.headers) as resp:
                if resp.status_code != 200:
                    raise ConnectionError(f"oanda stream HTTP {resp.status_code}")
                for line in resp.iter_lines():
                    q = self.parse_line(line)
                    if q is not None:
                        yield q


class TwelveDataPriceStream(QuoteStream):
    """Twelve Data WebSocket (paid plans). Only events carrying both bid and ask are stored."""

    source = "twelvedata"

    def __init__(self, api_key: str, instruments: Iterable[str], data_mode: str = "live", connect=None):
        self.api_key = api_key
        self.instruments = list(instruments)
        self.data_mode = data_mode
        self._connect = connect  # injectable for tests

    def parse_message(self, raw: str) -> Quote | None:
        msg = json.loads(raw)
        if msg.get("event") != "price" or msg.get("bid") is None or msg.get("ask") is None:
            return None
        try:
            ts = datetime.fromtimestamp(int(msg["timestamp"]), tz=timezone.utc)
            return Quote(
                instrument=msg["symbol"],
                source=self.source,
                ts=ts,
                bid=float(msg["bid"]),
                ask=float(msg["ask"]),
                data_mode=self.data_mode,
            )
        except (KeyError, ValueError, TypeError):
            return None

    def iter_quotes(self) -> Iterator[Quote]:
        if self._connect is None:
            import websocket  # websocket-client

            ws = websocket.create_connection(
                f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}", timeout=30
            )
        else:
            ws = self._connect()
        try:
            ws.send(json.dumps({"action": "subscribe", "params": {"symbols": ",".join(self.instruments)}}))
            while True:
                raw = ws.recv()
                if not raw:
                    return
                q = self.parse_message(raw)
                if q is not None:
                    yield q
        finally:
            try:
                ws.close()
            except Exception:  # pragma: no cover
                pass


class StreamRecorder:
    """Consume a :class:`QuoteStream`, persisting quotes in batches and reconnecting on failure."""

    def __init__(
        self,
        stream: QuoteStream,
        sink: Callable[[list[Quote]], int],
        on_error: Callable[[str], None] | None = None,
        batch_size: int = 200,
        flush_seconds: float = 5.0,
        max_backoff: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.stream = stream
        self.sink = sink
        self.on_error = on_error
        self.batch_size = batch_size
        self.flush_seconds = flush_seconds
        self.max_backoff = max_backoff
        self.sleep = sleep
        self.reconnects = 0
        self.stored = 0

    def run(self, stop: threading.Event, max_reconnects: int | None = None) -> None:
        backoff = 1.0
        while not stop.is_set():
            buffer: list[Quote] = []
            last_flush = time.monotonic()
            try:
                for quote in self.stream.iter_quotes():
                    buffer.append(quote)
                    backoff = 1.0  # healthy connection resets the backoff
                    if len(buffer) >= self.batch_size or time.monotonic() - last_flush >= self.flush_seconds:
                        self.stored += self.sink(buffer)
                        buffer, last_flush = [], time.monotonic()
                    if stop.is_set():
                        break
                message = "stream ended"
            except Exception as exc:  # network errors, protocol errors
                message = redact(f"stream error: {exc}")
            if buffer:
                self.stored += self.sink(buffer)
            if stop.is_set():
                return
            self.reconnects += 1
            log.warning("%s %s; reconnecting in %.0fs", self.stream.source, message, backoff)
            if self.on_error:
                self.on_error(message)
            if max_reconnects is not None and self.reconnects > max_reconnects:
                return
            self.sleep(backoff)
            backoff = min(self.max_backoff, backoff * 2)
