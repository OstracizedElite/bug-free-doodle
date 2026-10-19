-- PIT database schema (SQLite dialect; written to port to PostgreSQL with minimal change).
--
-- Conventions
--   * Timestamps are ISO-8601 UTC text ('YYYY-MM-DDTHH:MM:SSZ').
--   * *_min columns hold integer epoch minutes for fast range scans.
--   * Forecast, feature-snapshot, target, LLM-call and paper-transaction rows are
--     IMMUTABLE: UPDATE and DELETE are rejected by triggers. Outcomes live in separate,
--     append-only evaluation tables linked by forecast_id.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ============================================================ market data
CREATE TABLE IF NOT EXISTS candles (
    instrument       TEXT    NOT NULL,
    source           TEXT    NOT NULL,
    ts_min           INTEGER NOT NULL,          -- candle OPEN time, epoch minutes (1-minute candles)
    ts               TEXT    NOT NULL,
    open             REAL    NOT NULL,          -- mid prices unless quote_convention says otherwise
    high             REAL    NOT NULL,
    low              REAL    NOT NULL,
    close            REAL    NOT NULL,
    bid_open         REAL, bid_high REAL, bid_low REAL, bid_close REAL,
    ask_open         REAL, ask_high REAL, ask_low REAL, ask_close REAL,
    spread           REAL,                      -- closing spread, USD/oz
    spread_source    TEXT    NOT NULL,          -- observed | derived | assumed
    volume           REAL,
    quote_convention TEXT    NOT NULL,          -- e.g. mid_from_bid_ask, provider_mid, broker_cfd_mid, synthetic_mid
    data_mode        TEXT    NOT NULL,          -- synthetic | historical | delayed | live
    quality          TEXT    NOT NULL DEFAULT 'ok',   -- ok | suspect
    quality_notes    TEXT,
    ingested_at      TEXT    NOT NULL,
    PRIMARY KEY (instrument, source, ts_min)
);

CREATE TABLE IF NOT EXISTS quotes (
    instrument  TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    ts_ms       INTEGER NOT NULL,
    ts          TEXT    NOT NULL,
    bid         REAL    NOT NULL,
    ask         REAL    NOT NULL,
    mid         REAL    NOT NULL,
    spread      REAL    NOT NULL,
    data_mode   TEXT    NOT NULL,
    received_at TEXT    NOT NULL,
    PRIMARY KEY (instrument, source, ts_ms)
);

CREATE TABLE IF NOT EXISTS data_gaps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument      TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    gap_start_min   INTEGER NOT NULL,
    gap_start       TEXT    NOT NULL,
    gap_end         TEXT    NOT NULL,           -- exclusive
    missing_minutes INTEGER NOT NULL,
    detected_at     TEXT    NOT NULL,
    UNIQUE (instrument, source, gap_start_min)
);

CREATE TABLE IF NOT EXISTS data_quality_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument  TEXT NOT NULL,
    source      TEXT NOT NULL,
    ts          TEXT,
    event_type  TEXT NOT NULL,                  -- duplicate | rejected | suspect | stale | gap | recovered
    severity    TEXT NOT NULL,                  -- info | warning | error
    details     TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_dq_created ON data_quality_events (created_at);

CREATE TABLE IF NOT EXISTS feed_status (
    instrument       TEXT NOT NULL,
    source           TEXT NOT NULL,
    status           TEXT NOT NULL,             -- ok | stale | down | closed | unknown
    data_mode        TEXT,
    last_candle_ts   TEXT,
    last_success_at  TEXT,
    last_error_at    TEXT,
    last_error       TEXT,
    latency_seconds  REAL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (instrument, source)
);

CREATE TABLE IF NOT EXISTS api_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    component   TEXT NOT NULL,                  -- provider | llm | scheduler | ...
    provider    TEXT,
    endpoint    TEXT,
    status_code INTEGER,
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_api_errors_created ON api_errors (created_at);

CREATE TABLE IF NOT EXISTS api_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    component   TEXT NOT NULL,
    provider    TEXT NOT NULL,
    endpoint    TEXT,
    units       REAL NOT NULL DEFAULT 1,        -- requests or credits
    cost_usd    REAL NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_api_usage_created ON api_usage (created_at);

-- ============================================================ configuration & snapshots
CREATE TABLE IF NOT EXISTS config_versions (
    config_version TEXT PRIMARY KEY,
    scope          TEXT NOT NULL,               -- forecast | paper
    config_json    TEXT NOT NULL,
    first_seen_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id   TEXT PRIMARY KEY,             -- sha256 of the canonical snapshot payload
    instrument    TEXT NOT NULL,
    as_of_ts      TEXT NOT NULL,
    input_hash    TEXT NOT NULL,                -- sha256 of the raw candle window used
    features_json TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

-- ============================================================ forecasts (immutable)
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id          TEXT PRIMARY KEY,
    origin               TEXT    NOT NULL,      -- live | backtest
    run_id               TEXT    NOT NULL DEFAULT '',  -- backtest run id ('' for live)
    data_mode            TEXT    NOT NULL,      -- synthetic | historical | delayed | live
    instrument           TEXT    NOT NULL,
    provider             TEXT    NOT NULL,
    quote_convention     TEXT    NOT NULL,
    model_type           TEXT    NOT NULL,      -- deterministic | llm_assisted
    parent_forecast_id   TEXT REFERENCES forecasts(forecast_id),
    model_version        TEXT    NOT NULL,
    config_version       TEXT    NOT NULL,
    slot_ts              TEXT    NOT NULL,      -- scheduled slot
    slot_min             INTEGER NOT NULL,
    as_of_ts             TEXT    NOT NULL,      -- data cut-off: only candles closed at/before this
    created_at           TEXT    NOT NULL,      -- wall-clock time the record was produced
    start_ts             TEXT    NOT NULL,      -- evaluation starts here (>= creation, minute aligned)
    start_min            INTEGER NOT NULL,
    expiry_ts            TEXT    NOT NULL,
    expiry_min           INTEGER NOT NULL,
    horizon_minutes      INTEGER NOT NULL,
    reference_price      REAL,
    reference_price_type TEXT    NOT NULL,
    bid                  REAL,
    ask                  REAL,
    spread               REAL,
    spread_source        TEXT,
    direction            TEXT    NOT NULL,      -- UP | DOWN | WAIT
    candidate_direction  TEXT    NOT NULL,      -- UP | DOWN | NONE (sign of score, even under WAIT)
    composite_score      REAL,
    components_json      TEXT,
    target_price         REAL,
    target_distance      REAL,                  -- USD per troy ounce
    target_atr_multiple  REAL,
    stop_price           REAL,                  -- hypothetical stop used for executable evaluation
    atr_1m               REAL,
    sigma_1m             REAL,
    p_hit                REAL,
    p_hit_method         TEXT,
    p_hit_formula        TEXT,
    confidence           TEXT    NOT NULL,      -- high | medium | low | none
    regime               TEXT,
    vol_regime           TEXT,
    trend_regime         TEXT,
    session              TEXT,
    wait_reasons_json    TEXT    NOT NULL DEFAULT '[]',
    primary_wait_reason  TEXT,
    llm_provider         TEXT,
    llm_model            TEXT,
    llm_prompt_version   TEXT,
    llm_call_id          TEXT,
    snapshot_id          TEXT REFERENCES feature_snapshots(snapshot_id),
    content_hash         TEXT    NOT NULL,      -- sha256 of the canonical record (tamper evidence)
    UNIQUE (origin, run_id, instrument, slot_min, horizon_minutes, model_type, config_version)
);
CREATE INDEX IF NOT EXISTS ix_forecasts_slot ON forecasts (origin, run_id, slot_min);
CREATE INDEX IF NOT EXISTS ix_forecasts_expiry ON forecasts (expiry_min);

CREATE TABLE IF NOT EXISTS forecast_targets (
    forecast_id  TEXT NOT NULL REFERENCES forecasts(forecast_id),
    label        TEXT NOT NULL,                 -- primary | atr_0.5 | ... | mirror
    direction    TEXT NOT NULL,                 -- UP | DOWN
    atr_multiple REAL,
    distance     REAL NOT NULL,
    price        REAL NOT NULL,
    is_primary   INTEGER NOT NULL DEFAULT 0,
    is_mirror    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (forecast_id, label)
);

-- ============================================================ evaluations (append-only)
CREATE TABLE IF NOT EXISTS forecast_evaluations (
    evaluation_id            TEXT PRIMARY KEY,
    forecast_id              TEXT NOT NULL REFERENCES forecasts(forecast_id),
    evaluator_version        TEXT NOT NULL,
    evaluated_at             TEXT NOT NULL,
    status                   TEXT NOT NULL,     -- hit | miss | indeterminate | not_applicable
    coverage                 REAL,
    n_candles                INTEGER,
    expected_minutes         INTEGER,
    first_touch_ts           TEXT,
    minutes_to_touch         INTEGER,
    target_first             TEXT,              -- yes | no | ambiguous | neither (target vs mirror)
    mfe                      REAL,
    mae                      REAL,
    mae_before_hit           REAL,
    mae_before_hit_ambiguous INTEGER,
    close_at_expiry          REAL,
    direction_correct        INTEGER,           -- expiry close beyond reference in forecast direction
    trade_outcome            TEXT,              -- tp | sl | expired | ambiguous | not_filled | not_applicable
    trade_entry_ts           TEXT,
    trade_entry_price        REAL,
    trade_exit_ts            TEXT,
    trade_exit_price         REAL,
    trade_pnl_per_oz         REAL,              -- USD/oz after spread and slippage (conservative)
    trade_ambiguous          INTEGER,
    details_json             TEXT,
    UNIQUE (forecast_id, evaluator_version)
);

CREATE TABLE IF NOT EXISTS target_evaluations (
    forecast_id       TEXT NOT NULL REFERENCES forecasts(forecast_id),
    evaluator_version TEXT NOT NULL,
    label             TEXT NOT NULL,
    hit               INTEGER,                  -- 1 | 0 | NULL (indeterminate)
    first_touch_ts    TEXT,
    minutes_to_touch  INTEGER,
    PRIMARY KEY (forecast_id, evaluator_version, label)
);

-- ============================================================ LLM
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id         TEXT PRIMARY KEY,
    forecast_id     TEXT,                       -- deterministic forecast that was reviewed
    created_at      TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    cache_key       TEXT NOT NULL,
    cached          INTEGER NOT NULL DEFAULT 0,
    input_json      TEXT NOT NULL,
    response_json   TEXT,
    status          TEXT NOT NULL,              -- ok | invalid_response | error | budget_exceeded | disabled
    error           TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL NOT NULL DEFAULT 0,
    latency_ms      REAL
);
CREATE INDEX IF NOT EXISTS ix_llm_calls_cache ON llm_calls (cache_key);
CREATE INDEX IF NOT EXISTS ix_llm_calls_created ON llm_calls (created_at);

-- ============================================================ scheduling & health
CREATE TABLE IF NOT EXISTS missed_slots (
    instrument   TEXT    NOT NULL,
    slot_min     INTEGER NOT NULL,
    slot_ts      TEXT    NOT NULL,
    detected_at  TEXT    NOT NULL,
    reason       TEXT    NOT NULL,
    PRIMARY KEY (instrument, slot_min)
);

CREATE TABLE IF NOT EXISTS scheduler_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job         TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,                  -- ok | error | skipped
    message     TEXT
);
CREATE INDEX IF NOT EXISTS ix_scheduler_runs ON scheduler_runs (job, started_at);

CREATE TABLE IF NOT EXISTS heartbeats (
    component  TEXT PRIMARY KEY,
    last_beat  TEXT NOT NULL,
    details    TEXT
);

-- ============================================================ research runs
CREATE TABLE IF NOT EXISTS research_declarations (
    declaration_id  TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    config_version  TEXT NOT NULL,
    holdout_start   TEXT NOT NULL,
    holdout_end     TEXT NOT NULL,
    hypothesis      TEXT NOT NULL,
    consumed_by_run TEXT                         -- set once: an OOS holdout can be used only once
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id          TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL,              -- running | completed | failed
    validation_type TEXT NOT NULL,              -- exploratory | out_of_sample
    declaration_id  TEXT REFERENCES research_declarations(declaration_id),
    start_ts        TEXT NOT NULL,
    end_ts          TEXT NOT NULL,
    data_source     TEXT NOT NULL,
    data_mode       TEXT NOT NULL,
    config_version  TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    notes           TEXT,
    summary_json    TEXT
);

-- ============================================================ paper trading (separate from forecasts)
CREATE TABLE IF NOT EXISTS paper_accounts (
    account_id         TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    origin             TEXT NOT NULL,           -- live | backtest
    run_id             TEXT NOT NULL DEFAULT '',
    config_version     TEXT NOT NULL,
    config_json        TEXT NOT NULL,
    base_currency      TEXT NOT NULL,
    initial_balance    REAL NOT NULL,
    created_at         TEXT NOT NULL,
    last_processed_min INTEGER,
    state_json         TEXT NOT NULL            -- mutable simulator state (balance, peaks, halts)
);

CREATE TABLE IF NOT EXISTS paper_positions (
    position_id     TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES paper_accounts(account_id),
    forecast_id     TEXT NOT NULL,
    direction       TEXT NOT NULL,
    units           REAL NOT NULL,
    entry_ts        TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    stop_price      REAL NOT NULL,
    target_price    REAL NOT NULL,
    expiry_ts       TEXT NOT NULL,
    margin_usd      REAL NOT NULL,
    fx_rate_entry   REAL,
    status          TEXT NOT NULL,              -- open | closed
    exit_ts         TEXT,
    exit_price      REAL,
    exit_reason     TEXT,                       -- tp | sl | expiry | closeout | ambiguous_sl
    gross_pnl_usd   REAL,
    costs_usd       REAL,
    financing_usd   REAL,
    net_pnl_usd     REAL,
    net_pnl_base    REAL,                       -- in account currency; NULL if FX unavailable
    fx_rate_exit    REAL,
    ambiguous       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_paper_positions_account ON paper_positions (account_id, status);

CREATE TABLE IF NOT EXISTS paper_transactions (
    txn_id        TEXT PRIMARY KEY,
    account_id    TEXT NOT NULL REFERENCES paper_accounts(account_id),
    seq           INTEGER NOT NULL,
    ts            TEXT NOT NULL,
    txn_type      TEXT NOT NULL,                -- open | close | financing | closeout | skip | halt | resume
    position_id   TEXT,
    forecast_id   TEXT,
    amount_usd    REAL,
    amount_base   REAL,
    balance_after REAL,
    details_json  TEXT,
    UNIQUE (account_id, seq)
);

CREATE TABLE IF NOT EXISTS paper_equity (
    account_id   TEXT    NOT NULL REFERENCES paper_accounts(account_id),
    ts_min       INTEGER NOT NULL,
    ts           TEXT    NOT NULL,
    balance      REAL    NOT NULL,
    unrealised   REAL    NOT NULL,
    equity       REAL    NOT NULL,
    margin_used  REAL    NOT NULL,
    drawdown     REAL    NOT NULL,
    open_positions INTEGER NOT NULL,
    PRIMARY KEY (account_id, ts_min)
);

-- ============================================================ immutability triggers
CREATE TRIGGER IF NOT EXISTS trg_forecasts_no_update BEFORE UPDATE ON forecasts
BEGIN SELECT RAISE(ABORT, 'forecasts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_forecasts_no_delete BEFORE DELETE ON forecasts
BEGIN SELECT RAISE(ABORT, 'forecasts are immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_targets_no_update BEFORE UPDATE ON forecast_targets
BEGIN SELECT RAISE(ABORT, 'forecast targets are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_targets_no_delete BEFORE DELETE ON forecast_targets
BEGIN SELECT RAISE(ABORT, 'forecast targets are immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_snapshots_no_update BEFORE UPDATE ON feature_snapshots
BEGIN SELECT RAISE(ABORT, 'feature snapshots are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_snapshots_no_delete BEFORE DELETE ON feature_snapshots
BEGIN SELECT RAISE(ABORT, 'feature snapshots are immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_evals_no_update BEFORE UPDATE ON forecast_evaluations
BEGIN SELECT RAISE(ABORT, 'evaluations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_evals_no_delete BEFORE DELETE ON forecast_evaluations
BEGIN SELECT RAISE(ABORT, 'evaluations are append-only'); END;

CREATE TRIGGER IF NOT EXISTS trg_llm_no_update BEFORE UPDATE ON llm_calls
BEGIN SELECT RAISE(ABORT, 'llm call records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS trg_llm_no_delete BEFORE DELETE ON llm_calls
BEGIN SELECT RAISE(ABORT, 'llm call records are immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_txn_no_update BEFORE UPDATE ON paper_transactions
BEGIN SELECT RAISE(ABORT, 'paper transactions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_txn_no_delete BEFORE DELETE ON paper_transactions
BEGIN SELECT RAISE(ABORT, 'paper transactions are append-only'); END;

-- ============================================================ views
CREATE VIEW IF NOT EXISTS v_forecast_status AS
SELECT f.*,
       e.status            AS outcome_status,
       e.evaluated_at      AS evaluated_at,
       e.evaluator_version AS evaluator_version,
       e.first_touch_ts, e.minutes_to_touch, e.target_first, e.mfe, e.mae,
       e.mae_before_hit, e.mae_before_hit_ambiguous, e.close_at_expiry, e.direction_correct,
       e.trade_outcome, e.trade_pnl_per_oz, e.trade_ambiguous, e.coverage
FROM forecasts f
LEFT JOIN forecast_evaluations e
       ON e.forecast_id = f.forecast_id
      AND e.evaluated_at = (SELECT MAX(e2.evaluated_at) FROM forecast_evaluations e2
                            WHERE e2.forecast_id = f.forecast_id);
