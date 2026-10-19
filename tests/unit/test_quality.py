from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pit.config.settings import MarketConfig, SanityConfig
from pit.data.market_calendar import MarketCalendar
from pit.data.models import complete_bid_ask
from pit.data.quality import detect_gaps, feed_state, timestamps_to_epoch_min, validate_candles
from tests.helpers import MONDAY_10_UTC, make_candles

SANITY = SanityConfig()


def test_timestamp_normalisation_variants():
    s = pd.Series(["2026-01-05T10:00:00Z", "2026-01-05T12:01:00+02:00", "2026-01-05 10:02:00", "2026-01-05T10:03:30Z"])
    minutes, aligned = timestamps_to_epoch_min(s)
    assert list(minutes[:3] - MONDAY_10_UTC) == [0, 1, 2]
    assert list(aligned) == [True, True, True, False]
    ms = pd.Series([(MONDAY_10_UTC + 5) * 60_000])
    minutes, aligned = timestamps_to_epoch_min(ms)
    assert minutes[0] == MONDAY_10_UTC + 5 and aligned[0]
    # Naive timestamps in a non-UTC timezone.
    minutes, _ = timestamps_to_epoch_min(pd.Series(["2026-01-05 05:00:00"]), timezone="America/New_York")
    assert minutes[0] == MONDAY_10_UTC


def test_spread_completion_labels_source():
    df = pd.DataFrame(
        {
            "ts_min": [1, 2, 3],
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.5, 100.5, 100.5],
            "bid_close": [100.4, None, None],
            "ask_close": [100.6, None, None],
            "spread": [None, 0.4, None],
        }
    )
    out = complete_bid_ask(df, default_spread=0.3)
    assert list(out["spread_source"]) == ["observed", "derived", "assumed"]
    assert np.allclose(out["spread"], [0.2, 0.4, 0.3])
    assert np.isclose(out.loc[1, "bid_close"], 100.3) and np.isclose(out.loc[1, "ask_close"], 100.7)
    assert np.isclose(out.loc[2, "ask_high"], 101.15)


def test_duplicates_identical_and_conflicting():
    df = make_candles([2000, 2001, 2002])
    dup = df.iloc[[1]].copy()
    conflict = df.iloc[[2]].copy()
    conflict["close"] = 2002.5
    conflict["high"] = 2003.0
    batch = pd.concat([df, dup, conflict], ignore_index=True)
    clean, report = validate_candles(batch, SANITY, 0.3)
    assert len(clean) == 3
    assert report.duplicates == 2
    assert report.conflicting_duplicates == 1
    assert clean["close"].tolist() == [2000, 2001, 2002]  # first write kept


def test_invalid_candles_rejected():
    df = make_candles([2000, 2001, 2002, 2003, 2004])
    df.loc[1, "high"] = 1990.0  # high below open/close
    df.loc[2, "close"] = -5.0
    df.loc[3, "bid_close"] = df.loc[3, "ask_close"] + 1  # crossed market
    df.loc[4, "low"] = df.loc[4, "high"] + 1
    clean, report = validate_candles(df, SANITY, 0.3)
    reasons = dict(report.rejected)
    assert len(clean) == 1
    assert reasons[MONDAY_10_UTC + 1] == "high_below_open_close"
    assert reasons[MONDAY_10_UTC + 2] in ("non_positive_price", "low_above_open_close")
    assert reasons[MONDAY_10_UTC + 3] == "bid_above_ask"
    assert MONDAY_10_UTC + 4 in reasons


def test_price_jump_and_wide_spread_flagged_suspect():
    df = make_candles([2000, 2000.5, 2200.0, 2200.5])
    df.loc[3, ["bid_close", "ask_close"]] = [2180.0, 2221.0]
    clean, report = validate_candles(df, SANITY, 0.3, prev_close=2000.0)
    assert clean["quality"].tolist() == ["ok", "ok", "suspect", "suspect"]
    assert clean.loc[2, "quality_notes"] == "price_jump"
    assert "wide_spread" in clean.loc[3, "quality_notes"]
    assert len(report.suspect) == 2


def test_incomplete_candles_dropped():
    df = make_candles([2000, 2001, 2002])
    clean, report = validate_candles(df, SANITY, 0.3, now_min=MONDAY_10_UTC + 2)
    assert clean["ts_min"].tolist() == [MONDAY_10_UTC, MONDAY_10_UTC + 1]
    assert report.incomplete == 1


def test_gap_detection_ignores_closed_market():
    cal = MarketCalendar(MarketConfig())
    start = MONDAY_10_UTC
    present = np.array([start, start + 1, start + 4, start + 5, start + 9])
    gaps = detect_gaps(present, cal, start, start + 10)
    assert [(g.start_min - start, g.end_min - start, g.missing_minutes) for g in gaps] == [(2, 4, 2), (6, 9, 3)]
    # Friday 21:58 UTC -> Sunday 23:02 UTC: only the 2 Friday and 2 Sunday minutes are expected.
    from pit.common.timeutils import to_epoch_min

    fri = to_epoch_min("2026-01-09T21:58:00Z")
    sun = to_epoch_min("2026-01-11T23:00:00Z")
    present = np.array([fri, fri + 1, sun, sun + 1])
    assert detect_gaps(present, cal, fri, sun + 2) == []


def test_feed_state():
    cal = MarketCalendar(MarketConfig())
    now = datetime(2026, 1, 5, 10, 10, 30, tzinfo=timezone.utc)
    last = MONDAY_10_UTC + 9  # completes at 10:10:00 -> 30 s old
    assert feed_state(last, now, cal, 180) == ("ok", 30.0)
    assert feed_state(MONDAY_10_UTC, now, cal, 180)[0] == "stale"
    saturday = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
    assert feed_state(MONDAY_10_UTC, saturday, cal, 180)[0] == "closed"
    assert feed_state(None, now, cal, 180)[0] == "unknown"
