# Architecture

```
                       ┌──────────────────────────── pit/scheduler/runner.py ───────────────────────────┐
                       │ tick(now): ingest ─▶ forecast (per slot) ─▶ evaluate ─▶ paper ─▶ heartbeat     │
                       └─────┬────────────────────┬──────────────────────┬───────────────┬──────────────┘
                             │                    │                      │               │
  providers (pit/data)       ▼                    ▼                      ▼               ▼
  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌───────────────┐ ┌────────────────┐
  │ synthetic    │   │ IngestionSvc  │   │ ForecastService  │   │ EvaluationSvc │ │ PaperTrading   │
  │ csv          │──▶│ validate      │   │  MarketFrames    │   │  touch/MFE/   │ │  engine (bid/  │
  │ twelvedata   │   │ store, gaps   │   │  FeatureEngine   │   │  bracket on   │ │  ask, margin,  │
  │ oanda (RO)   │   │ feed status   │   │  Deterministic   │   │  bid/ask      │ │  financing,    │
  │ massive      │   └──────┬────────┘   │  Forecaster      │   └──────┬────────┘ │  FX, limits)   │
  └──────────────┘          │            │  (+ LLMReviewer) │          │          └──────┬─────────┘
                            ▼            └────────┬─────────┘          ▼                 ▼
                    ┌─────────────────────────────────────────────────────────────────────────────┐
                    │ SQLite (pit/database): candles · quotes · gaps · feed_status · forecasts      │
                    │ (immutable) · targets · snapshots · evaluations (append-only) · llm_calls     │
                    │ · paper_* · backtest_runs · declarations · scheduler_runs · heartbeats        │
                    └───────────────────────────────┬───────────────────────────────────────────────┘
                                                    ▼
                         analytics (pit/analytics) ─▶ Streamlit dashboard (pit/dashboard) · CLI (pit/app)
```

## Packages

| Package | Responsibility |
|---|---|
| `pit/config` | Pydantic settings, YAML layering, config-version hashing |
| `pit/common` | UTC/epoch-minute helpers, logging |
| `pit/data` | domain models, market calendar, provider interface and adapters, validation, ingestion, tick streaming |
| `pit/features` | look-ahead-safe resampling (`MarketFrames`), indicator primitives, the feature engine |
| `pit/forecasting` | forecast record model, deterministic engine (scoring, targets, P-HIT, WAIT rules), events, live service, optional LLM reviewer |
| `pit/evaluation` | outcome evaluation (touch, MFE/MAE, executable bracket, ambiguity) and its service |
| `pit/backtesting` | historical replay that shares the live forecasting path |
| `pit/paper_trading` | virtual account engine and services |
| `pit/analytics` | metrics, calibration, bootstrap intervals, breakdowns, comparisons |
| `pit/database` | schema, connection management, repositories (all SQL lives here) |
| `pit/scheduler` | tick-based scheduler, lock, recovery |
| `pit/dashboard` | Streamlit app, views, charts |
| `pit/app` | application context, CLI, health aggregation |

## Key design decisions

1. **Epoch minutes internally.** Timestamps are integers (minutes since 1970), so
   look-ahead checks reduce to integer comparisons (`bar_end ≤ as_of`), with no timezone
   or datetime-resolution pitfalls. ISO-8601 UTC strings are used at the storage and UI
   boundaries.
2. **One forecasting code path.** Live forecasting and replay call the same
   `DeterministicForecaster.forecast(frames, slot, …)`. A test asserts they produce
   identical records for the same data.
3. **Fixed lookback windows.** A forecast depends only on a bounded window of completed
   bars, so it is reproducible from any superset of history.
4. **Immutability at the database level.** Triggers reject UPDATE and DELETE on
   forecasts, targets, snapshots, evaluations, LLM calls and paper transactions. Content
   hashes provide tamper evidence. Outcomes live in separate append-only tables.
5. **Explicit provenance.** Every candle carries its `source`, `data_mode` (synthetic,
   historical, delayed or live), `quote_convention` and `spread_source` (observed,
   derived or assumed). Every forecast carries its model version, config version,
   provider and data mode.
6. **No fabricated data.** Missing minutes become recorded gaps. A stale feed makes new
   forecasts WAIT. AUD values are never estimated without an observed rate.
7. **LLM as an optional filter.** It reviews a stored deterministic forecast and writes
   a separate linked record. It is never required, and it is budget-capped and logged.
8. **Deterministic scheduler.** `tick(now)` is a pure function of the clock and the
   database state. Recovery behaviour (missed slots, outages, restarts) is tested with a
   simulated clock.

## Moving to PostgreSQL later

All SQL lives in `pit/database/*_repo.py` and uses `?` placeholders with mostly
standard SQL. A Postgres backend needs:

* a connection class with the same interface as `Database`, using `%s` placeholders;
* `INSERT … ON CONFLICT DO NOTHING` in place of `INSERT OR IGNORE`;
* `GENERATED … AS IDENTITY` in place of `AUTOINCREMENT`;
* the immutability triggers rewritten as PL/pgSQL functions that `RAISE EXCEPTION`;
* `BOOLEAN` and `TIMESTAMPTZ` types where SQLite uses `INTEGER` and `TEXT`.

The schema is otherwise portable.
