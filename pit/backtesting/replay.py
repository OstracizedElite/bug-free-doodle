"""Historical replay (backtesting).

At each simulated slot ``T`` the runner:
  1. exposes only candles completed at or before ``T`` (``MarketFrames.select``);
  2. computes features and the forecast with the *same* code path as live forecasting;
  3. stores the forecast (``origin='backtest'``, ``created_at`` = the simulated clock);
then, after the replay, evaluates every forecast on the candles that followed it and
optionally simulates the paper account.

Guarantees and caveats:
* No look-ahead: features use completed bars only; evaluation starts at ``start_ts``.
* Parameters are never optimised automatically. Comparing several predeclared
  configurations is *exploratory*; an ``out_of_sample`` run requires a research declaration
  made beforehand, whose config hash must match and whose holdout window must contain the
  run window. Each declaration can be consumed by exactly one run.
* Overlapping forecasts are expected (15-minute interval, 60-minute horizon); analytics
  account for the dependence.
* LLM review is disabled in replay unless explicitly allowed: language models may have
  seen historical prices in training (look-ahead bias).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from pit.analytics import metrics
from pit.app.context import AppContext
from pit.common.timeutils import ensure_utc, from_epoch_min, iso, iso_min, parse_ts, to_epoch_min
from pit.data.market_calendar import MarketCalendar
from pit.evaluation.evaluator import CandleStore
from pit.evaluation.service import EvaluationService
from pit.features.frames import MarketFrames
from pit.forecasting.deterministic.engine import DeterministicForecaster
from pit.forecasting.models import ForecastContext, ForecastResult

log = logging.getLogger(__name__)


class BacktestValidationError(ValueError):
    pass


@dataclass
class BacktestResult:
    run_id: str
    forecasts: int
    evaluated: int
    seconds: float
    summary: dict = field(default_factory=dict)
    paper_account_id: str | None = None


class BacktestRunner:
    def __init__(self, ctx: AppContext, source: str | None = None, store_snapshots: bool = False):
        # The configuration under test is the context's: build the context from the YAML being
        # evaluated (``pit --config candidate.yaml backtest ...``) so every component agrees.
        self.ctx = ctx
        self.settings = ctx.settings
        self.source = source or ctx.provider.name
        self.calendar = MarketCalendar(self.settings.market)
        self.store_snapshots = store_snapshots

    def _validate_oos(self, declaration_id: str | None, start_min: int, end_min: int) -> dict:
        if not declaration_id:
            raise BacktestValidationError("out_of_sample runs require a research declaration (pit declare)")
        decl = self.ctx.research.declaration(declaration_id)
        if decl is None:
            raise BacktestValidationError(f"unknown declaration {declaration_id}")
        if decl["consumed_by_run"]:
            raise BacktestValidationError(
                f"declaration {declaration_id} was already used by run {decl['consumed_by_run']}; "
                "re-running a holdout makes it in-sample"
            )
        if decl["config_version"] != self.settings.forecast_config_version:
            raise BacktestValidationError(
                f"config {self.settings.forecast_config_version} differs from declared {decl['config_version']}"
            )
        if start_min < to_epoch_min(decl["holdout_start"]) or end_min > to_epoch_min(decl["holdout_end"]):
            raise BacktestValidationError("run window lies outside the declared holdout period")
        return decl

    def slots(self, start_min: int, end_min: int) -> list[int]:
        interval = self.settings.forecasting.interval_minutes
        first = start_min + (-start_min % interval)
        return [m for m in range(first, end_min, interval) if self.calendar.is_valid_forecast_slot(m)]

    def run(
        self,
        start: datetime | str,
        end: datetime | str,
        name: str = "backtest",
        validation_type: str = "exploratory",
        declaration_id: str | None = None,
        notes: str | None = None,
        paper: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> BacktestResult:
        t0 = time.time()
        s = self.settings
        start_min = to_epoch_min(parse_ts(start) if isinstance(start, str) else ensure_utc(start))
        end_min = to_epoch_min(parse_ts(end) if isinstance(end, str) else ensure_utc(end))
        if end_min <= start_min:
            raise BacktestValidationError("end must be after start")
        if validation_type not in ("exploratory", "out_of_sample"):
            raise BacktestValidationError("validation_type must be exploratory or out_of_sample")
        if validation_type == "out_of_sample":
            self._validate_oos(declaration_id, start_min, end_min)

        instrument = s.instrument.symbol
        max_h = max(s.forecasting.horizons)
        warmup = s.data.history_days * 1440
        candles = self.ctx.candles.load(instrument, self.source, start_min - warmup, end_min + max_h + 1)
        if candles.empty:
            raise BacktestValidationError(f"no candles stored for {instrument} from source {self.source!r}")
        data_mode = str(candles["data_mode"].iloc[-1])
        quote_convention = str(candles["quote_convention"].iloc[-1])

        run_id = self.ctx.research.create_run(
            name, validation_type, iso_min(start_min), iso_min(end_min), self.source, data_mode,
            s.forecast_config_version, s.forecast_config_dict(), declaration_id, notes,
        )
        if validation_type == "out_of_sample" and not self.ctx.research.consume_declaration(declaration_id, run_id):
            self.ctx.research.finish_run(run_id, "failed", {"error": "declaration already consumed"})
            raise BacktestValidationError("declaration already consumed")
        self.ctx.ops.register_config(s.forecast_config_version, "forecast", s.forecast_config_dict())
        try:
            frames = MarketFrames(candles, s.features.timeframes)
            forecaster = DeterministicForecaster(s, self.calendar)
            fctx = ForecastContext(origin="backtest", data_mode=data_mode, provider=self.source,
                                   quote_convention=quote_convention, instrument=instrument, run_id=run_id)
            slots = self.slots(start_min, end_min)
            reviewer = None
            if s.llm.enabled and s.llm.allow_in_backtest:
                from pit.forecasting.llm.reviewer import LLMReviewer

                reviewer = LLMReviewer.from_context(self.ctx)
                log.warning("LLM review enabled in backtest: results are NOT credible evidence (look-ahead risk)")
            batch: list[ForecastResult] = []
            stored = 0
            for k, slot in enumerate(slots):
                results = forecaster.forecast(frames, slot, fctx, created_at=from_epoch_min(slot))
                if reviewer is not None:
                    reviewer.record_clock = lambda slot=slot: from_epoch_min(slot)
                    results += [r for r in (reviewer.review(res) for res in results) if r is not None]
                batch.extend(results)
                if len(batch) >= 500:
                    stored += self.ctx.forecasts.insert_many(batch, store_snapshot=self.store_snapshots)
                    batch = []
                if progress and k % 200 == 0:
                    progress(k, len(slots))
            stored += self.ctx.forecasts.insert_many(batch, store_snapshot=self.store_snapshots)

            # Evaluate on the stored candles (same evaluator as live).
            store = CandleStore(candles)
            pending = self.ctx.forecasts.pending_evaluation(
                s.evaluation.version, end_min + max_h + 1, origin="backtest", run_id=run_id, limit=10**9
            )
            evaluated = EvaluationService(self.ctx).evaluate(pending, store)

            df = self.ctx.forecasts.frame(origin="backtest", run_id=run_id)
            summary = {
                "deterministic": metrics.summarise(df[df["model_type"] == "deterministic"], s.forecasting.interval_minutes),
                "mirror_edge": metrics.mirror_edge(self.ctx.forecasts.target_evaluations("backtest", run_id),
                                                   s.forecasting.interval_minutes),
                "slots": len(slots),
                "data_mode": data_mode,
                "validation_type": validation_type,
                "warning": None if data_mode != "synthetic" else
                "SYNTHETIC DATA - results are not evidence about real markets",
                "llm_in_backtest": bool(s.llm.enabled and s.llm.allow_in_backtest),
            }
            account_id = None
            if paper and s.paper.enabled:
                from pit.paper_trading.service import run_paper_backtest

                fx = self.ctx.candles.load(s.instrument.fx_symbol, self.source, start_min - 1440, end_min + max_h + 1)
                forecasts = [self.ctx.forecasts.get(fid) for fid in df.loc[df["direction"] != "WAIT", "forecast_id"]]
                in_window = candles[candles["ts_min"] >= start_min]
                account_id, events, engine = run_paper_backtest(self.ctx, run_id, in_window, fx, [f for f in forecasts if f])
                summary["paper"] = {
                    "account_id": account_id,
                    "final_balance": round(engine.state.balance, 2),
                    "unconverted_usd": round(engine.state.unconverted_usd, 2),
                    "trades_closed": len(events.closed),
                    "drawdown_halted": engine.state.drawdown_halted,
                }
            self.ctx.research.finish_run(run_id, "completed", summary)
        except Exception as exc:
            self.ctx.research.finish_run(run_id, "failed", {"error": str(exc)})
            raise
        return BacktestResult(run_id, stored, evaluated, round(time.time() - t0, 2), summary, account_id)
