from pit.common.timeutils import to_epoch_min
from pit.config.settings import MarketConfig
from pit.data.market_calendar import MarketCalendar


def cal(**kw):
    return MarketCalendar(MarketConfig(**kw))


def m(ts: str) -> int:
    return to_epoch_min(ts)


def test_weekday_open_and_daily_break_winter():
    c = cal()
    # Winter (EST, UTC-5): daily break 17:00-18:00 NY = 22:00-23:00 UTC.
    assert c.is_open(m("2026-01-06T21:59:00Z"))
    assert not c.is_open(m("2026-01-06T22:00:00Z"))
    assert not c.is_open(m("2026-01-06T22:59:00Z"))
    assert c.is_open(m("2026-01-06T23:00:00Z"))


def test_daily_break_shifts_with_daylight_saving():
    c = cal()
    # Summer (EDT, UTC-4): break is 21:00-22:00 UTC.
    assert not c.is_open(m("2026-07-07T21:00:00Z"))
    assert c.is_open(m("2026-07-07T22:00:00Z"))
    assert c.is_open(m("2026-07-07T20:59:00Z"))


def test_weekend_closed():
    c = cal()
    assert c.is_open(m("2026-01-09T21:59:00Z"))  # Fri 16:59 NY
    assert not c.is_open(m("2026-01-09T22:00:00Z"))  # Fri 17:00 NY
    assert not c.is_open(m("2026-01-10T12:00:00Z"))  # Saturday
    assert not c.is_open(m("2026-01-11T22:59:00Z"))  # Sun 17:59 NY
    assert c.is_open(m("2026-01-11T23:00:00Z"))  # Sun 18:00 NY


def test_open_minutes_count_for_a_full_week():
    c = cal()
    start, end = m("2026-01-11T00:00:00Z"), m("2026-01-18T00:00:00Z")
    # Sun 18:00 -> Fri 17:00 = 119 hours minus 4 daily breaks (Mon-Thu) = 115 hours.
    assert c.count_open_minutes(start, end) == 115 * 60


def test_holiday_closes_whole_local_date():
    c = cal(holidays=["2026-01-07"])
    assert not c.is_open(m("2026-01-07T15:00:00Z"))
    assert c.is_open(m("2026-01-08T15:00:00Z"))


def test_session_labels_and_valid_slots():
    c = cal()
    assert c.session_label(m("2026-01-06T03:00:00Z")) == "asia"
    assert c.session_label(m("2026-01-06T23:30:00Z")) == "asia"
    assert c.session_label(m("2026-01-06T08:00:00Z")) == "london"
    assert c.session_label(m("2026-01-06T13:00:00Z")) == "overlap"
    assert c.session_label(m("2026-01-06T17:00:00Z")) == "new_york"
    # Slot right at the reopen has no just-completed candle -> invalid.
    assert not c.is_valid_forecast_slot(m("2026-01-06T23:00:00Z"))
    assert c.is_valid_forecast_slot(m("2026-01-06T23:15:00Z"))
    assert c.first_closed_minute(m("2026-01-06T21:30:00Z"), m("2026-01-06T22:30:00Z")) == m("2026-01-06T22:00:00Z")
