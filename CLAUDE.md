# CLAUDE.md — working on PIT

PIT is an XAU/USD forecasting **research** platform (Python 3.12, SQLite, Streamlit).
Read `docs/architecture.md` and `docs/methodology.md` before changing forecasting logic.

## Commands

```bash
.venv/bin/python -m pytest                    # full suite (~3 min, 140 tests); add -m "not integration" for unit tests only
.venv/bin/python -m pit --help                # CLI
.venv/bin/python -m pit --db var/dev.db demo  # synthetic end-to-end run
.venv/bin/streamlit run pit/dashboard/app.py
```

## Invariants (do not break)

1. **No live trading.** Never add order, trade or position endpoints, or any brokerage
   write access. Provider adapters are read-only.
2. **No look-ahead.** Features use only bars with `end <= as_of` (`MarketFrames.select`).
   Evaluation starts at `start_ts`. Keep `tests/integration/test_backtest_and_pipeline.py`
   (poisoned future, live-vs-replay equality) passing.
3. **Fixed lookback windows.** Feature results must not depend on how much older history
   is loaded.
4. **Immutability.** Never UPDATE or DELETE forecasts, targets, snapshots, evaluations,
   LLM calls or paper transactions. Outcomes go into the append-only tables.
5. **No fabricated data.** Never fill missing prices, guess spreads without labelling
   them `assumed`, or invent FX rates. A stale feed means WAIT.
6. **Never back-date.** A late slot is recorded as missed. `created_at` is real time
   (simulated only for `origin='backtest'`).
7. **Honest labelling.** Synthetic data is always labelled synthetic. P-HIT is always
   described as an uncalibrated heuristic. Never claim profitability.
8. **No ML in V1** and no automatic parameter optimisation.
9. **No costs without approval.** Do not enable paid providers or LLM calls, or add
   pricing you have not verified (an unpriced model is never called).
10. **Bump `pit.MODEL_VERSION`** when code changes could alter a forecast for identical
    inputs and config. Parameters are versioned automatically via the config hash.

## Conventions

* Time is integer epoch minutes internally (`pit/common/timeutils.py`), with ISO-8601
  UTC `...Z` strings at the boundaries.
* All SQL lives in `pit/database/*_repo.py` with `?` placeholders (to stay portable to
  Postgres).
* New parameters go in `config/default.yaml` **and** the Pydantic model in
  `pit/config/settings.py` (unknown keys are rejected).
* Tests use deterministic data (`tests/helpers.py`) with hand-computed expected values;
  provider HTTP is mocked with `httpx.MockTransport`; the scheduler is tested with a
  simulated clock.
* Dashboard colours: polarity blue `#2a78d6` (UP) and red `#e34948` (DOWN), always with
  ▲/▼ shapes; one y-axis per chart; every chart has a table view (`docs/dashboard.md`).
