"""REST provider adapters against mocked HTTP transports (no network access)."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from pit.common.timeutils import to_epoch_min
from pit.data.providers.base import AuthenticationError, ProviderError, ProviderUnavailableError
from pit.data.providers.csv_file import CsvProvider
from pit.data.providers.http import HttpJsonClient, RateLimiter, redact
from pit.data.providers.massive import MassiveProvider
from pit.data.providers.oanda import OandaProvider
from pit.data.providers.twelvedata import TwelveDataProvider

T0 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 5, 10, 5, tzinfo=timezone.utc)


class FakeOps:
    def __init__(self):
        self.errors, self.usage = [], []

    def log_api_error(self, component, message, provider=None, endpoint=None, status_code=None):
        self.errors.append((component, message, provider, endpoint, status_code))

    def log_api_usage(self, component, provider, endpoint=None, units=1, cost_usd=0.0):
        self.usage.append((component, provider, endpoint))


def no_sleep(_):
    pass


def test_twelvedata_parses_series_and_assumes_no_bid_ask():
    seen = {}

    def handler(request: httpx.Request):
        seen.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "meta": {"symbol": "XAU/USD"},
                "values": [
                    {"datetime": "2026-01-05 10:00:00", "open": "2000.1", "high": "2001", "low": "1999.5", "close": "2000.5"},
                    {"datetime": "2026-01-05 10:01:00", "open": "2000.5", "high": "2002", "low": "2000", "close": "2001.5"},
                ],
                "status": "ok",
            },
        )

    p = TwelveDataProvider("KEY", transport=httpx.MockTransport(handler), sleep=no_sleep)
    df = p.fetch_candles("XAU/USD", T0, T1)
    assert seen["symbol"] == "XAU/USD" and seen["interval"] == "1min" and seen["timezone"] == "UTC"
    assert df["ts_min"].tolist() == [to_epoch_min(T0), to_epoch_min(T0) + 1]
    assert df["close"].tolist() == [2000.5, 2001.5]
    assert df["bid_close"].isna().all()  # spreads are filled later and labelled 'assumed'


def test_twelvedata_no_data_is_empty_not_error_but_other_errors_raise():
    responses = iter(
        [
            {"code": 400, "message": "No data is available on the specified dates.", "status": "error"},
            {"code": 400, "message": "Invalid symbol", "status": "error"},
        ]
    )
    p = TwelveDataProvider("KEY", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=next(responses))))
    assert p.fetch_candles("XAU/USD", T0, T1).empty
    with pytest.raises(ProviderError):
        p.fetch_candles("XAU/USD", T0, T1)


def test_oanda_uses_bid_ask_and_drops_incomplete_candles():
    def handler(request: httpx.Request):
        assert request.headers["Authorization"] == "Bearer TOKEN"
        assert request.url.path == "/v3/instruments/XAU_USD/candles"
        assert request.url.params["price"] == "MBA"
        block = lambda o: {"o": str(o), "h": str(o + 1), "l": str(o - 1), "c": str(o + 0.5)}  # noqa: E731
        return httpx.Response(
            200,
            json={
                "candles": [
                    {"complete": True, "volume": 5, "time": "2026-01-05T10:00:00.000000000Z",
                     "mid": block(2000.0), "bid": block(1999.85), "ask": block(2000.15)},
                    {"complete": False, "volume": 2, "time": "2026-01-05T10:01:00.000000000Z",
                     "mid": block(2001.0), "bid": block(2000.85), "ask": block(2001.15)},
                ]
            },
        )

    p = OandaProvider("TOKEN", transport=httpx.MockTransport(handler), sleep=no_sleep)
    df = p.fetch_candles("XAU/USD", T0, T1)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["bid_close"] == pytest.approx(2000.35) and row["ask_close"] == pytest.approx(2000.65)


def test_massive_follows_pagination_and_injects_key():
    calls = []

    def handler(request: httpx.Request):
        calls.append(request.url)
        assert request.url.params["apiKey"] == "KEY"
        t = to_epoch_min(T0) * 60_000
        if "cursor" not in request.url.params:
            return httpx.Response(
                200,
                json={"status": "OK", "results": [{"t": t, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 3}],
                      "next_url": "https://api.massive.com/v2/aggs/ticker/C:XAUUSD/range/1/minute/a/b?cursor=abc"},
            )
        return httpx.Response(200, json={"status": "OK", "results": [{"t": t + 60_000, "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 1}]})

    p = MassiveProvider("KEY", transport=httpx.MockTransport(handler), sleep=no_sleep)
    df = p.fetch_candles("XAU/USD", T0, T1)
    assert len(calls) == 2
    assert df["close"].tolist() == [1.5, 1.8]
    assert "/v2/aggs/ticker/C:XAUUSD/range/1/minute/" in str(calls[0])


def test_auth_failure_not_retried_and_logged_without_secret():
    ops = FakeOps()
    count = {"n": 0}

    def handler(request):
        count["n"] += 1
        return httpx.Response(401, text="bad key")

    client = HttpJsonClient("x", "https://example.test", 600, transport=httpx.MockTransport(handler), ops=ops, sleep=no_sleep)
    with pytest.raises(AuthenticationError):
        client.get_json("/q", {"apikey": "SECRET123"})
    assert count["n"] == 1
    assert ops.errors and all("SECRET123" not in str(e) for e in ops.errors)


def test_rate_limit_and_server_errors_retried_with_backoff():
    sleeps = []
    seq = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(503), httpx.Response(200, json={"ok": 1})])
    client = HttpJsonClient(
        "x", "https://example.test", 6000, transport=httpx.MockTransport(lambda r: next(seq)), sleep=sleeps.append
    )
    assert client.get_json("/q") == {"ok": 1}
    assert sleeps[0] == 7.0  # honours Retry-After
    assert sleeps[1] == 4.0  # 2 * 2**1 exponential backoff


def test_persistent_server_error_raises_after_max_retries():
    client = HttpJsonClient(
        "x", "https://example.test", 6000, max_retries=2,
        transport=httpx.MockTransport(lambda r: httpx.Response(500)), sleep=no_sleep,
    )
    with pytest.raises(ProviderUnavailableError):
        client.get_json("/q")


def test_network_error_retried():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] < 3:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=[1])

    client = HttpJsonClient("x", "https://example.test", 6000, transport=httpx.MockTransport(handler), sleep=no_sleep)
    assert client.get_json("/q") == [1]
    assert state["n"] == 3


def test_rate_limiter_token_bucket():
    clock = {"t": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    rl = RateLimiter(2, clock=lambda: clock["t"], sleep=sleep)  # 2 per minute
    assert rl.acquire() == 0 and rl.acquire() == 0
    waited = rl.acquire()
    assert waited == pytest.approx(30.0)


def test_redact():
    assert redact("https://x?a=1&apikey=abc&b=2") == "https://x?a=1&apikey=***&b=2"


def test_csv_provider(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text(
        "Time,Open,High,Low,Close,Bid,Ask\n"
        "2026-01-05 05:00:00,2000,2001,1999,2000.5,2000.4,2000.6\n"
        "2026-01-05 05:01:00,2000.5,2002,2000,2001.5,2001.4,2001.6\n"
        "2026-01-05 05:01:30,2001.5,2002,2001,2001.7,2001.6,2001.8\n"
    )
    p = CsvProvider(path, timezone="America/New_York")
    df = p.fetch_candles("XAU/USD", T0, T1)
    assert df["ts_min"].tolist() == [to_epoch_min(T0), to_epoch_min(T0) + 1]
    assert p.misaligned_rows == 1
    assert df["bid_close"].tolist() == [2000.4, 2001.4]
