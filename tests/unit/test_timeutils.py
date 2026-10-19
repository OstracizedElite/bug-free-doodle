from datetime import datetime, timedelta, timezone

from pit.common.timeutils import (
    ceil_minute,
    floor_to_interval,
    from_epoch_min,
    is_aligned,
    iso,
    parse_ts,
    to_epoch_min,
)


def test_parse_normalises_to_utc():
    assert parse_ts("2026-01-05T10:00:00Z") == datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    assert parse_ts("2026-01-05T12:00:00+02:00") == datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    # Naive timestamps are interpreted as UTC.
    assert parse_ts("2026-01-05 10:00:00") == datetime(2026, 1, 5, 10, tzinfo=timezone.utc)


def test_epoch_minute_round_trip():
    dt = datetime(2026, 1, 5, 10, 7, 33, tzinfo=timezone.utc)
    m = to_epoch_min(dt)
    assert from_epoch_min(m) == datetime(2026, 1, 5, 10, 7, tzinfo=timezone.utc)
    assert iso(from_epoch_min(m)) == "2026-01-05T10:07:00Z"


def test_ceil_and_intervals():
    dt = datetime(2026, 1, 5, 10, 7, 1, tzinfo=timezone.utc)
    assert ceil_minute(dt) == datetime(2026, 1, 5, 10, 8, tzinfo=timezone.utc)
    exact = datetime(2026, 1, 5, 10, 7, tzinfo=timezone.utc)
    assert ceil_minute(exact) == exact
    assert floor_to_interval(dt, 15) == datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    assert is_aligned(datetime(2026, 1, 5, 10, 15, tzinfo=timezone.utc), 15)
    assert not is_aligned(datetime(2026, 1, 5, 10, 16, tzinfo=timezone.utc), 15)
    assert not is_aligned(datetime(2026, 1, 5, 10, 15, 1, tzinfo=timezone.utc), 15)
    assert to_epoch_min(dt + timedelta(minutes=1)) - to_epoch_min(dt) == 1
