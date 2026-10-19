"""Unattended operation: a deterministic, tick-based scheduler.

``Scheduler.tick(now)`` runs every job that is due at ``now`` and is safe to call at any
frequency; ``run_forever`` simply calls it about once a second. Because all timing is
derived from the ``now`` argument, recovery behaviour is fully testable with a simulated
clock.

Jobs
  ingest     every ``ingest_interval_seconds``: fetch completed candles (all instruments)
  forecast   once per slot, ``forecast_delay_seconds`` after the slot boundary, during
             valid market sessions; data are synced first so the slot's final candle is in
  evaluate   every ``evaluation_interval_seconds``
  paper      every ``paper_interval_seconds``
  heartbeat  every ``heartbeat_interval_seconds``
  missed     at start-up and hourly: valid slots without a forecast are recorded (never
             back-filled)

Reliability
  * every job runs inside its own error boundary and is logged to ``scheduler_runs``;
  * duplicate forecasts are impossible (unique slot key) even if two processes run;
  * a slot later than ``grace_seconds`` is recorded as missed, never forecast late;
  * an advisory lock file prevents two schedulers on one database;
  * configuration files are watched and reloaded between ticks (new config version).
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from pit.app.context import AppContext
from pit.common.timeutils import ensure_utc, iso_min, to_epoch_min, utcnow
from pit.config.settings import DEFAULT_CONFIG_PATH, LOCAL_CONFIG_PATH, load_settings
from pit.data.ingestion import IngestionService
from pit.data.market_calendar import MarketCalendar
from pit.evaluation.service import EvaluationService
from pit.forecasting.service import ForecastService
from pit.paper_trading.service import PaperTradingService

log = logging.getLogger(__name__)


class SchedulerLockError(RuntimeError):
    pass


class _FileLock:
    """Advisory exclusive lock (fcntl on POSIX; best-effort elsewhere)."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        try:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # pragma: no cover - non-POSIX
            pass
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise SchedulerLockError(f"another scheduler holds {self.path}") from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(str(os.getpid()))
        self._fh.flush()

    def release(self) -> None:
        if self._fh is not None:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):  # pragma: no cover
                pass
            self._fh.close()
            self._fh = None


@dataclass
class _JobState:
    last_run: float | None = None
    failures: int = 0


@dataclass
class TickReport:
    now: str
    ran: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    forecast: dict | None = None


class Scheduler:
    def __init__(
        self,
        ctx: AppContext,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], None] = time.sleep,
        use_lock: bool = True,
        watch_config: bool = True,
    ):
        self.ctx = ctx
        self.clock = clock
        self.sleep = sleep
        self.use_lock = use_lock
        self.watch_config = watch_config
        self._lock = _FileLock(ctx.db.path.with_suffix(".scheduler.lock"))
        self._jobs: dict[str, _JobState] = {}
        self._attempted_slots: set[int] = set()
        self._config_mtime = self._config_stamp()
        self.stop_event = threading.Event()
        self._build_services()

    # ------------------------------------------------------------------ wiring
    def _build_services(self) -> None:
        s = self.ctx.settings
        self.calendar = MarketCalendar(s.market)
        self.ingestion = IngestionService(self.ctx)
        reviewer = None
        if s.llm.enabled:
            from pit.forecasting.llm.reviewer import LLMReviewer

            reviewer = LLMReviewer.from_context(self.ctx)
        self.forecasts = ForecastService(self.ctx, llm_reviewer=reviewer)
        self.evaluation = EvaluationService(self.ctx)
        self.paper = PaperTradingService(self.ctx)
        self.intervals = {
            "ingest": s.scheduler.ingest_interval_seconds,
            "evaluate": s.scheduler.evaluation_interval_seconds,
            "paper": s.scheduler.paper_interval_seconds,
            "heartbeat": s.scheduler.heartbeat_interval_seconds,
            "missed": 3600,
            "maintenance": 24 * 3600,
        }

    @staticmethod
    def _config_stamp() -> tuple:
        return tuple(p.stat().st_mtime if p.exists() else 0.0 for p in (DEFAULT_CONFIG_PATH, LOCAL_CONFIG_PATH))

    def _maybe_reload_config(self) -> None:
        if not self.watch_config:
            return
        stamp = self._config_stamp()
        if stamp == self._config_mtime:
            return
        self._config_mtime = stamp
        try:
            new = load_settings()
        except Exception as exc:  # invalid edit: keep running on the old configuration
            log.error("configuration change rejected (keeping current config): %s", exc)
            self.ctx.ops.log_api_error("scheduler", f"config reload failed: {exc}")
            return
        old_version = self.ctx.settings.forecast_config_version
        self.ctx = AppContext.create(settings=new, db_path=self.ctx.db.path)
        self._build_services()
        log.info("configuration reloaded: forecast config %s -> %s", old_version, new.forecast_config_version)

    # ------------------------------------------------------------------ jobs
    def _due(self, job: str, now_ts: float) -> bool:
        st = self._jobs.setdefault(job, _JobState())
        return st.last_run is None or now_ts - st.last_run >= self.intervals[job]

    def _run(self, job: str, fn: Callable[[], object], now: datetime, report: TickReport, record: bool = True):
        st = self._jobs.setdefault(job, _JobState())
        st.last_run = now.timestamp()
        run_id = self.ctx.ops.start_job(job, now) if record else None
        try:
            result = fn()
        except Exception as exc:  # error boundary: a failing job never stops the scheduler
            st.failures += 1
            log.exception("job %s failed", job)
            report.errors[job] = str(exc)
            if run_id is not None:
                self.ctx.ops.finish_job(run_id, "error", str(exc), now)
            return None
        st.failures = 0
        report.ran.append(job)
        if run_id is not None:
            self.ctx.ops.finish_job(run_id, "ok", _summarise(result), now)
        return result

    def startup(self, now: datetime | None = None) -> None:
        now = ensure_utc(now or self.clock())
        if self.use_lock:
            self._lock.acquire()
        self.ctx.ops.heartbeat("scheduler", self._hb_details("starting"), now)
        report = TickReport(now=now.isoformat())
        self._run("ingest", lambda: [r.as_dict() for r in self.ingestion.sync_all(now)], now, report)
        self._run("missed", lambda: [iso_min(m) for m in self.forecasts.detect_missed_slots(now)], now, report)
        log.info("scheduler started (provider=%s, config=%s)", self.ctx.settings.data.provider,
                 self.ctx.settings.forecast_config_version)

    def shutdown(self) -> None:
        try:
            self.ctx.ops.heartbeat("scheduler", self._hb_details("stopped"), self.clock())
        finally:
            self._lock.release()

    def _hb_details(self, state: str) -> dict:
        return {"state": state, "pid": os.getpid(), "provider": self.ctx.settings.data.provider,
                "config_version": self.ctx.settings.forecast_config_version}

    def tick(self, now: datetime | None = None) -> TickReport:
        now = ensure_utc(now or self.clock())
        ts = now.timestamp()
        report = TickReport(now=now.isoformat())
        self._maybe_reload_config()
        s = self.ctx.settings

        # Forecast slot handling comes first: a due slot gets a fresh sync immediately before it.
        slot = self.forecasts.current_slot(now)
        delay = s.scheduler.forecast_delay_seconds
        if slot not in self._attempted_slots and ts >= slot * 60 + delay and self.calendar.is_valid_forecast_slot(slot):
            self._attempted_slots.add(slot)
            self._run("ingest", lambda: [r.as_dict() for r in self.ingestion.sync_all(now)], now, report)
            outcome = self._run("forecast", lambda: self.forecasts.run_slot(slot, now), now, report)
            if outcome is not None:
                report.forecast = outcome.as_dict()
            # keep the attempted set small
            self._attempted_slots = {m for m in self._attempted_slots if m >= slot - 24 * 60}
        elif self._due("ingest", ts):
            self._run("ingest", lambda: [r.as_dict() for r in self.ingestion.sync_all(now)], now, report)

        if self._due("evaluate", ts):
            self._run("evaluate", lambda: self.evaluation.evaluate_due(now), now, report)
        if s.paper.enabled and self._due("paper", ts):
            self._run("paper", lambda: self.paper.process(now), now, report)
        if self._due("missed", ts):
            self._run("missed", lambda: [iso_min(m) for m in self.forecasts.detect_missed_slots(now)], now, report)
        if self._due("maintenance", ts):
            cutoff = iso_min(to_epoch_min(now) - 30 * 1440)
            self._run("maintenance", lambda: {"pruned_job_runs": self.ctx.ops.prune_job_runs(cutoff)}, now, report)
        if self._due("heartbeat", ts):
            self._jobs["heartbeat"].last_run = ts
            self.ctx.ops.heartbeat("scheduler", self._hb_details("running"), now)
        return report

    # ------------------------------------------------------------------ loop
    def run_forever(self, max_ticks: int | None = None) -> None:
        self._install_signal_handlers()
        self.startup()
        ticks = 0
        try:
            while not self.stop_event.is_set():
                self.tick()
                ticks += 1
                if max_ticks is not None and ticks >= max_ticks:
                    break
                # Wake shortly after the next whole second.
                now = time.time()
                self.stop_event.wait(max(0.05, 1.0 - (now % 1.0) + 0.01))
        finally:
            self.shutdown()
            log.info("scheduler stopped")

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def handler(signum, frame):
            log.info("signal %s received: stopping after the current job", signum)
            self.stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # pragma: no cover
                pass


def _summarise(result: object) -> str:
    if result is None:
        return ""
    if hasattr(result, "as_dict"):
        result = result.as_dict()
    text = str(result)
    return text[:1500]
