from datetime import datetime, timezone

import numpy as np
import pytest

from pit.common.timeutils import to_epoch_min
from pit.config.settings import MarketConfig, SyntheticProviderConfig
from pit.data.market_calendar import MarketCalendar
from pit.data.providers.base import InstrumentNotSupportedError
from pit.data.providers.synthetic import SyntheticProvider

CFG = SyntheticProviderConfig(anchor="2026-01-01T00:00:00Z")
CAL = MarketCalendar(MarketConfig())


def provider(**kw):
    cfg = CFG.model_copy(update=kw)
    return SyntheticProvider(cfg, CAL, clock=lambda: datetime(2026, 3, 1, tzinfo=timezone.utc))


def test_deterministic_regardless_of_request_order():
    start = to_epoch_min("2026-01-12T00:00:00Z")
    a = provider().generate("XAU/USD", start, start + 3 * 1440)
    p = provider()
    p.generate("XAU/USD", start + 2 * 1440, start + 3 * 1440)  # request later window first
    b = p.generate("XAU/USD", start, start + 3 * 1440)
    assert a.equals(b)
    c = provider(seed=999).generate("XAU/USD", start, start + 3 * 1440)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_only_open_market_minutes_and_consistent_ohlc():
    start, end = to_epoch_min("2026-01-09T00:00:00Z"), to_epoch_min("2026-01-12T00:00:00Z")
    df = provider().generate("XAU/USD", start, end)
    assert CAL.open_mask(df["ts_min"].to_numpy()).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["ask_close"] > df["bid_close"]).all()
    assert (df["data_mode"] == "synthetic").all()
    assert (df["spread_source"] == "observed").all()


def test_never_returns_future_candles():
    clock_time = datetime(2026, 1, 6, 12, 30, 20, tzinfo=timezone.utc)
    p = SyntheticProvider(CFG, CAL, clock=lambda: clock_time)
    df = p.fetch_candles("XAU/USD", datetime(2026, 1, 6, 12, 0, tzinfo=timezone.utc), datetime(2026, 1, 7, tzinfo=timezone.utc))
    assert df["ts_min"].max() == to_epoch_min("2026-01-06T12:29:00Z")
    q = p.fetch_latest_quote("XAU/USD")
    assert q is not None and q.ask > q.bid


def test_missing_rate_drops_minutes_and_fx_supported():
    start = to_epoch_min("2026-01-13T00:00:00Z")
    full = provider().generate("XAU/USD", start, start + 1440)
    holey = provider(missing_rate=0.05).generate("XAU/USD", start, start + 1440)
    assert 0 < len(full) - len(holey) < 0.1 * len(full)
    fx = provider().generate("AUD/USD", start, start + 1440)
    assert 0.3 < fx["close"].mean() < 1.2
    with pytest.raises(InstrumentNotSupportedError):
        provider().generate("BTC/USD", start, start + 10)
