"""UTC time helpers.

Internally PIT represents minute timestamps as integer *epoch minutes* (minutes since
1970-01-01T00:00Z). Integer arithmetic avoids timezone and datetime-resolution pitfalls,
makes look-ahead checks trivial (``bar_end_min <= as_of_min``) and is fast in NumPy.
At storage and display boundaries timestamps are ISO-8601 UTC strings ending in ``Z``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Return ``dt`` as an aware UTC datetime. Naive datetimes are assumed to be UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_ts(value: str | datetime) -> datetime:
    """Parse an ISO-8601 string (``Z`` or offset suffix accepted) into an aware UTC datetime."""
    if isinstance(value, datetime):
        return ensure_utc(value)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    return ensure_utc(datetime.fromisoformat(text))


def iso(dt: datetime) -> str:
    """Canonical storage format: ``YYYY-MM-DDTHH:MM:SSZ`` (sub-second precision kept if present)."""
    dt = ensure_utc(dt)
    if dt.microsecond:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_epoch_min(dt: datetime | str) -> int:
    """Epoch minute containing ``dt`` (floor)."""
    d = parse_ts(dt)
    return int(d.timestamp() // 60)


def from_epoch_min(minute: int) -> datetime:
    return datetime.fromtimestamp(int(minute) * 60, tz=UTC)


def iso_min(minute: int) -> str:
    return iso(from_epoch_min(minute))


def floor_minute(dt: datetime) -> datetime:
    dt = ensure_utc(dt)
    return dt.replace(second=0, microsecond=0)


def ceil_minute(dt: datetime) -> datetime:
    dt = ensure_utc(dt)
    floored = floor_minute(dt)
    return floored if floored == dt else floored + timedelta(minutes=1)


def floor_to_interval(dt: datetime, minutes: int) -> datetime:
    m = to_epoch_min(dt)
    return from_epoch_min(m - (m % minutes))


def is_aligned(dt: datetime, minutes: int) -> bool:
    dt = ensure_utc(dt)
    return dt.second == 0 and dt.microsecond == 0 and to_epoch_min(dt) % minutes == 0
