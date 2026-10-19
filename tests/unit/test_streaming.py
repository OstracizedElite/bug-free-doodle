import json
import threading

from pit.data.streaming import OandaPriceStream, QuoteStream, StreamRecorder, TwelveDataPriceStream


def test_oanda_stream_parses_prices_and_ignores_heartbeats():
    s = OandaPriceStream("T", "ACC", ["XAU/USD"])
    price = json.dumps(
        {"type": "PRICE", "instrument": "XAU_USD", "time": "2026-01-05T10:00:01.123456789Z",
         "bids": [{"price": "2000.10"}], "asks": [{"price": "2000.40"}]}
    )
    q = s.parse_line(price)
    assert q.instrument == "XAU/USD" and q.bid == 2000.10 and abs(q.spread - 0.30) < 1e-9
    assert s.parse_line(json.dumps({"type": "HEARTBEAT", "time": "2026-01-05T10:00:05Z"})) is None
    assert s.parse_line("") is None


def test_twelvedata_stream_requires_bid_and_ask():
    s = TwelveDataPriceStream("K", ["XAU/USD"])
    assert s.parse_message(json.dumps({"event": "price", "symbol": "XAU/USD", "timestamp": 1767607200, "price": 2000})) is None
    q = s.parse_message(json.dumps({"event": "price", "symbol": "XAU/USD", "timestamp": 1767607200, "bid": 1999.9, "ask": 2000.2}))
    assert q is not None and q.mid == (1999.9 + 2000.2) / 2


def test_recorder_reconnects_with_backoff_and_flushes():
    oanda = OandaPriceStream("T", "ACC", ["XAU/USD"])
    line = json.dumps(
        {"type": "PRICE", "instrument": "XAU_USD", "time": "2026-01-05T10:00:01Z",
         "bids": [{"price": "2000.1"}], "asks": [{"price": "2000.4"}]}
    )

    class Fake(QuoteStream):
        source = "fake"

        def __init__(self):
            self.connects = 0

        def iter_quotes(self):
            self.connects += 1
            yield oanda.parse_line(line)
            raise ConnectionError("dropped")

    stored, sleeps, errors = [], [], []
    rec = StreamRecorder(Fake(), sink=lambda qs: stored.extend(qs) or len(qs), on_error=errors.append,
                         batch_size=100, sleep=sleeps.append)
    rec.run(threading.Event(), max_reconnects=3)
    assert len(stored) == 4  # buffered quote flushed after each drop
    assert sleeps == [1.0, 1.0, 1.0]  # backoff resets after a healthy (data-bearing) connection
    assert len(errors) == 4 and "dropped" in errors[0]
