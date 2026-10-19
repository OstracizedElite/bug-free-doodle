# Database

SQLite, file `var/pit.db` by default (`database.path` or `PIT_DATABASE_PATH`), WAL mode.
The schema is in `pit/database/schema.sql` and is applied idempotently on start-up.

## Tables

| Table | Mutability | Contents |
|---|---|---|
| `candles` | insert-only (first write wins) | 1-minute mid OHLC plus bid/ask OHLC, spread and `spread_source`, `quote_convention`, `data_mode`, `quality` |
| `quotes` | insert-only | tick bid/ask (streaming providers) |
| `data_gaps` | insert-only | runs of missing open-market minutes |
| `data_quality_events` | insert-only | duplicates, rejections, suspect candles, stale and recovered feeds |
| `feed_status` | upserted | current status per instrument and source |
| `api_errors`, `api_usage` | insert-only | provider and LLM errors, request counts and costs |
| `config_versions` | insert-only | full JSON of every configuration version seen |
| `feature_snapshots` | **immutable** (trigger) | the complete feature set behind a forecast, keyed by hash |
| `forecasts` | **immutable** (trigger) | every forecast (fields below) |
| `forecast_targets` | **immutable** (trigger) | primary, mirror and alternative target levels |
| `forecast_evaluations` | **append-only** (trigger) | one row per forecast per evaluator version |
| `target_evaluations` | insert-only | hit and timing per target level |
| `llm_calls` | **immutable** (trigger) | every LLM attempt: input, response, tokens, cost, latency, status |
| `missed_slots` | insert-only | slots with no forecast (never back-filled) |
| `scheduler_runs`, `heartbeats` | operational | job log (pruned after 30 days) and liveness |
| `research_declarations` | insert, consumed once | pre-registered out-of-sample tests |
| `backtest_runs` | status updated | replay metadata and summary |
| `paper_accounts`, `paper_positions` | simulator state | account state; positions are closed by update |
| `paper_transactions` | **append-only** (trigger) | every simulated open, close, financing charge, skip, halt and resume |
| `paper_equity` | insert or replace | equity curve points |

## Forecast record

Each forecast stores:

* **Identity:** `forecast_id`, `origin` (live or backtest), `run_id`.
* **Provenance:** `data_mode`, `instrument`, `provider`, `quote_convention`,
  `model_type` (deterministic or llm_assisted), `parent_forecast_id`, `model_version`,
  `config_version`.
* **Time:** `slot_ts`, `as_of_ts` (data cut-off), `created_at`, `start_ts`, `expiry_ts`,
  `horizon_minutes`.
* **Prices:** `reference_price` and `reference_price_type`, `bid`, `ask`, `spread`,
  `spread_source`.
* **Signal:** `direction`, `candidate_direction`, `composite_score`, `components_json`
  (components, weights, conflict, P-HIT details).
* **Targets:** `target_price`, `target_distance` (USD/oz), `target_atr_multiple`,
  `stop_price`, `atr_1m`, `sigma_1m`.
* **Probability:** `p_hit`, `p_hit_method` (`uncalibrated_heuristic`), `p_hit_formula`.
* **Context:** `confidence`, `regime`, `vol_regime`, `trend_regime`, `session`.
* **WAIT:** `wait_reasons_json`, `primary_wait_reason`.
* **LLM:** `llm_provider`, `llm_model`, `llm_prompt_version`, `llm_call_id`.
* **Integrity:** `snapshot_id`, `content_hash`.

Outcome status and evaluation time are **not** columns of `forecasts`, because that
record never changes. The view `v_forecast_status` joins each forecast to its latest
evaluation (`outcome_status`, `evaluated_at`, touch timing, MFE/MAE, trade outcome and
P&L).

## Integrity guarantees

* `UNIQUE(origin, run_id, instrument, slot_min, horizon_minutes, model_type,
  config_version)`: one forecast per slot, model and configuration.
* Triggers abort any UPDATE or DELETE on immutable tables.
* `content_hash` = SHA-256 of the canonical record; `pit verify` recomputes every hash.
* `PRAGMA integrity_check` runs from `pit verify` and System Health.

## Backups

`pit backup` uses SQLite's online backup API, so a backup is consistent while the
scheduler runs.
