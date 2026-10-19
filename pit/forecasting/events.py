"""Manually maintained, verified economic-event restrictions (config/events.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pit.common.timeutils import to_epoch_min
from pit.config.settings import EventsConfig


@dataclass(frozen=True)
class EconomicEvent:
    name: str
    minute: int
    before: int
    after: int
    source: str = ""

    @property
    def window(self) -> tuple[int, int]:
        return self.minute - self.before, self.minute + self.after


class EventCalendar:
    def __init__(self, events: list[EconomicEvent] | None = None):
        self.events = sorted(events or [], key=lambda e: e.minute)

    @classmethod
    def from_file(cls, path: str | Path, cfg: EventsConfig) -> "EventCalendar":
        p = Path(path)
        if not p.exists():
            return cls([])
        data = yaml.safe_load(p.read_text("utf-8")) or {}
        events = []
        for item in data.get("events") or []:
            events.append(
                EconomicEvent(
                    name=str(item["name"]),
                    minute=to_epoch_min(str(item["time"])),
                    before=int(item.get("minutes_before", cfg.minutes_before)),
                    after=int(item.get("minutes_after", cfg.minutes_after)),
                    source=str(item.get("source", "")),
                )
            )
        return cls(events)

    def restricting(self, start_min: int, end_min: int) -> list[EconomicEvent]:
        """Events whose restricted window overlaps the forecast window ``[start, end)``."""
        return [e for e in self.events if e.window[0] < end_min and e.window[1] > start_min]
