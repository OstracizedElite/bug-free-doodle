# PIT — Predictive Intelligence Trading

An autonomous **XAU/USD (gold) forecasting and research platform**. Every 15 minutes it
collects prices, computes deterministic quantitative features, and issues an UP, DOWN or
WAIT forecast with a price target, a horizon and a heuristic hit probability. Each
forecast is recorded permanently *before* its outcome is known. PIT then evaluates it
against later prices, simulates a paper-trading account, and reports accuracy,
calibration and cost-adjusted expectancy in a dashboard.

> **Research and paper trading only.** There is no brokerage integration and PIT never
> places a real order. The forecasting method is a **testable hypothesis, not a proven
> edge**: nothing here should be read as evidence of profitability without forward-testing
> results (see [`docs/research_protocol.md`](docs/research_protocol.md)). Results on
> synthetic data are labelled as such everywhere and are never trading performance.

## What it does

| Area | Highlights |
|---|---|
| Market data | Provider-independent interface: synthetic, CSV, Twelve Data, OANDA (read-only), Massive/Polygon. Validates duplicates, gaps, sanity checks, stale feeds and rate limits. Never fabricates prices. Records where each spread came from |
| Features | Momentum, volatility, trend, market structure and regime over 1m to 12h, computed from **completed bars only** on fixed windows, so live and replay give identical results |
| Forecasts | Weighted composite score of seven normalised components, volatility-adjusted targets with a **mirror control target**, ten documented WAIT conditions, and a versioned **uncalibrated** P-HIT heuristic |
| Integrity | Immutable records (database triggers plus SHA-256 content hashes), feature snapshots, config versioning; missed slots are recorded and never back-dated |
| Evaluation | Mid-price target touches, timing, MFE/MAE, and **executable bid/ask bracket outcomes** with slippage. Intrabar ambiguity is never counted as a win |
| Backtesting | Same code path as live, no look-ahead (tested with poisoned future data), exploratory vs pre-registered out-of-sample runs |
| Paper trading | A$50k virtual account: bid/ask execution, risk-based sizing, 20:1 leverage, margin close-out, financing, daily-loss and drawdown halts, AUD conversion from observed FX only |
| Analytics | Edge vs mirror, Brier score and reliability diagrams, expectancy and profit factor, breakdowns, block-bootstrap intervals for overlapping forecasts |
| LLM (optional) | OpenAI or Anthropic reviewer that can only accept or reject deterministic signals; separate linked records; budget-capped, cached and fully logged; off by default |
| Operations | Tick-based scheduler with recovery, lock, heartbeats and hot config reload; Streamlit dashboard with eight pages; CLI |

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                     # run the test suite
.venv/bin/python -m pit demo --days 21         # synthetic data + a labelled synthetic backtest
.venv/bin/streamlit run pit/dashboard/app.py   # open http://localhost:8501
```

Run it continuously (synthetic "live" mode by default):

```bash
.venv/bin/python -m pit run                    # Ctrl+C to stop; restart-safe
```

Then pick the backtest run in the dashboard sidebar ("Forecast scope") to explore it.

## Common commands

```bash
pit ingest --days 100                  # backfill history from the configured provider
pit import-csv gold_1m.csv --source vendor --timezone UTC
pit forecast                           # forecast the current slot
pit backtest --start 2026-08-01 --end 2026-09-01 --ingest
pit declare --holdout-start 2026-09-01 --holdout-end 2026-10-01 --hypothesis "..."
pit backtest --start 2026-09-01 --end 2026-10-01 --validation out_of_sample --declaration decl_...
pit report --run bt_...                # analytics as JSON
pit status | pit verify | pit backup
```

The full list is in [`docs/operations.md`](docs/operations.md).

## Configuration

`config/default.yaml` defines every parameter: provider, sessions, feature windows,
weights, thresholds, targets, WAIT rules, P-HIT coefficients, evaluation, paper trading,
LLM, scheduler and costs. Overrides go in `config/local.yaml` (git-ignored; the
dashboard Settings page writes it after validation). Credentials go in `.env` only
(see `.env.example`).

Forecast-relevant parameters are hashed into a **config version** stored on every
forecast, so results from different settings are never mixed.

## Data providers

The default is synthetic data (no cost, no credentials). The provider comparison and
the recommendation (OANDA practice account for free bid/ask data, subject to approval)
are in [`docs/providers.md`](docs/providers.md). **No paid subscription, account or API
credit is used without explicit approval.**

## Project layout

```
pit/
  app/            CLI, application context, health
  config/         settings model, layering, version hashing
  common/         time utilities, logging
  data/           models, market calendar, providers/, validation, ingestion, streaming
  features/       resampling (completed bars), indicators, feature engine
  forecasting/    models, deterministic/ engine, llm/ reviewer, events, live service
  evaluation/     outcome evaluation and service
  backtesting/    historical replay
  paper_trading/  account engine and services
  analytics/      metrics, calibration, bootstrap, comparisons
  database/       schema.sql, connection, repositories
  scheduler/      tick-based scheduler
  dashboard/      Streamlit app, views, charts
config/           default.yaml, events.yaml
docs/             architecture, methodology, providers, operations, database, research protocol, dashboard
tests/            unit/ and integration/ (pytest)
scripts/          helper scripts
```

## Documentation

* [Architecture](docs/architecture.md)
* [Forecast methodology](docs/methodology.md): every formula and rule
* [Research protocol](docs/research_protocol.md): what counts as evidence
* [Data providers](docs/providers.md)
* [Operations](docs/operations.md): install, run, deploy, monitor, back up
* [Database](docs/database.md)
* [Dashboard](docs/dashboard.md)

## Status

Version 1.0 implements all seven development phases. Live market data needs an approved
provider. Until then the platform runs end to end on synthetic data, which is clearly
labelled throughout.
