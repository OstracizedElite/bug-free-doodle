"""Gold trading calendar.

Spot gold / XAU CFDs trade almost continuously from Sunday evening to Friday afternoon
New York time, with a daily one-hour maintenance break. The calendar is configured in the
exchange timezone (``America/New_York`` by default) so daylight-saving transitions are
handled correctly. A minute ``m`` (epoch minutes) is *open* when trading is possible
throughout ``[m, m+1)``.

Rules (all configurable):
  * closed from ``week_close`` (Fri 17:00) until ``week_open`` (Sun 18:00);
  * closed during the daily break ``[daily_break_start, daily_break_end)``;
  * closed on configured holiday dates (exchange-local calendar dates).
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pit.common.timeutils import from_epoch_min
from pit.config.settings import MarketConfig

_DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _parse_hhmm(text: str) -> int:
    hh, mm = text.strip().split(":")
    return int(hh) * 60 + int(mm)


def _parse_day_time(text: str) -> tuple[int, int]:
    day, hhmm = text.strip().split()
    return _DAYS[day.lower()[:3]], _parse_hhmm(hhmm)


class MarketCalendar:
    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.timezone)
        self.open_day, self.open_tod = _parse_day_time(cfg.week_open)
        self.close_day, self.close_tod = _parse_day_time(cfg.week_close)
        self.break_start = _parse_hhmm(cfg.daily_break_start)
        self.break_end = _parse_hhmm(cfg.daily_break_end)
        self.holidays = {date.fromisoformat(d) for d in cfg.holidays}
        self.sessions = {k: (int(v[0]), int(v[1])) for k, v in cfg.sessions_utc.items()}
        self._masks: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ open/closed
    def _day_mask(self, day: int) -> np.ndarray:
        """Open mask for the 1440 minutes of UTC day ``day`` (epoch_min // 1440), cached."""
        mask = self._masks.get(day)
        if mask is None:
            mask = self._compute_mask(np.arange(day * 1440, (day + 1) * 1440, dtype="int64"))
            mask.setflags(write=False)
            if len(self._masks) > 20_000:
                self._masks.clear()
            self._masks[day] = mask
        return mask

    def open_mask(self, minutes: np.ndarray) -> np.ndarray:
        """Vectorised: True where the epoch minute is an open trading minute."""
        minutes = np.asarray(minutes, dtype="int64")
        if minutes.size == 0:
            return np.zeros(0, dtype=bool)
        days = minutes // 1440
        d0, d1 = int(days.min()), int(days.max())
        if d0 == d1:
            return self._day_mask(d0)[minutes - d0 * 1440].copy()
        table = np.concatenate([self._day_mask(d) for d in range(d0, d1 + 1)]) if d1 - d0 < 4000 else None
        if table is not None:
            return table[minutes - d0 * 1440]
        return self._compute_mask(minutes)

    def _compute_mask(self, minutes: np.ndarray) -> np.ndarray:
        local = pd.DatetimeIndex(pd.to_datetime(minutes * 60, unit="s", utc=True)).tz_convert(self.tz)
        dow = np.asarray(local.dayofweek)
        tod = np.asarray(local.hour) * 60 + np.asarray(local.minute)
        # Minutes since the start of the trading week, measured from Monday 00:00 local.
        week_pos = dow * 1440 + tod
        open_pos = self.open_day * 1440 + self.open_tod
        close_pos = self.close_day * 1440 + self.close_tod
        if open_pos > close_pos:  # usual case: week opens Sunday, closes Friday
            in_week = (week_pos >= open_pos) | (week_pos < close_pos)
        else:
            in_week = (week_pos >= open_pos) & (week_pos < close_pos)
        if self.break_start != self.break_end:
            if self.break_start < self.break_end:
                in_break = (tod >= self.break_start) & (tod < self.break_end)
            else:
                in_break = (tod >= self.break_start) | (tod < self.break_end)
        else:
            in_break = np.zeros_like(in_week)
        mask = in_week & ~in_break
        if self.holidays:
            local_dates = np.asarray(local.date)
            hol = np.isin(local_dates, np.array(sorted(self.holidays), dtype=object))
            mask &= ~hol
        return mask

    def is_open(self, minute: int) -> bool:
        day = int(minute) // 1440
        return bool(self._day_mask(day)[int(minute) - day * 1440])

    def open_minutes(self, start_min: int, end_min: int) -> np.ndarray:
        """Open epoch minutes in ``[start_min, end_min)``."""
        if end_min <= start_min:
            return np.zeros(0, dtype="int64")
        minutes = np.arange(start_min, end_min, dtype="int64")
        return minutes[self.open_mask(minutes)]

    def count_open_minutes(self, start_min: int, end_min: int) -> int:
        if end_min <= start_min:
            return 0
        total = 0
        day = start_min // 1440
        while day * 1440 < end_min:
            lo = max(start_min, day * 1440) - day * 1440
            hi = min(end_min, (day + 1) * 1440) - day * 1440
            total += int(np.count_nonzero(self._day_mask(day)[lo:hi]))
            day += 1
        return total

    def first_closed_minute(self, start_min: int, end_min: int) -> int | None:
        """First closed minute in ``[start_min, end_min)``, or None if open throughout."""
        day = start_min // 1440
        while day * 1440 < end_min:
            lo = max(start_min, day * 1440) - day * 1440
            hi = min(end_min, (day + 1) * 1440) - day * 1440
            closed = np.flatnonzero(~self._day_mask(day)[lo:hi])
            if closed.size:
                return int(day * 1440 + lo + closed[0])
            day += 1
        return None

    def is_valid_forecast_slot(self, slot_min: int) -> bool:
        """A slot is valid when a candle has just completed (open at slot-1) and trading continues."""
        return self.is_open(slot_min - 1) and self.is_open(slot_min)

    # ------------------------------------------------------------------ labels
    def session_label(self, minute: int) -> str:
        return _session_for_hour(from_epoch_min(minute).hour, tuple(sorted(self.sessions.items())))

    def session_labels(self, minutes: np.ndarray) -> np.ndarray:
        hours = (np.asarray(minutes, dtype="int64") // 60) % 24
        key = tuple(sorted(self.sessions.items()))
        return np.array([_session_for_hour(int(h), key) for h in hours], dtype=object)

    def local_time(self, minute: int) -> datetime:
        return from_epoch_min(minute).astimezone(self.tz)


@lru_cache(maxsize=256)
def _session_for_hour(hour: int, sessions: tuple) -> str:
    for name, (start, end) in sessions:
        if start <= end:
            if start <= hour < end:
                return name
        elif hour >= start or hour < end:
            return name
    return "other"
