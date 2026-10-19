# Installation and operations

## 1. Install

Requires Python 3.11 or newer (3.12 recommended).

```bash
git clone <repo> pit && cd pit
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # or: pip install -r requirements.txt
cp .env.example .env                        # only fill in services you have approved
.venv/bin/python -m pit init-db
.venv/bin/python -m pytest                  # all tests should pass
```

The default configuration uses **synthetic data** and needs no credentials and no
network access.

## 2. Day-to-day commands

| Command | Purpose |
|---|---|
| `pit demo --days 21` | load synthetic data and run a labelled synthetic backtest |
| `pit ingest --days 100` | backfill 1-minute history from the configured provider |
| `pit import-csv file.csv --source mydata --timezone UTC` | validate and import an approved CSV |
| `pit sync` | one incremental ingestion cycle |
| `pit forecast` | forecast the current slot (only within the grace window) |
| `pit evaluate` | evaluate forecasts whose horizon has passed |
| `pit paper` | advance the live paper account |
| `pit run` | **run the scheduler** (all of the above, unattended) |
| `pit run --once` | start-up plus one tick (cron mode) |
| `pit backtest --start … --end … [--ingest]` | historical replay |
| `pit declare --holdout-start … --holdout-end … --hypothesis …` | pre-register an OOS test |
| `pit report [--run RUN_ID]` | analytics as JSON |
| `pit status` | health summary |
| `pit verify` | database integrity and forecast content-hash verification |
| `pit backup [--to file]` | consistent online SQLite backup |
| `pit llm-preview` | show the exact LLM prompt (no API call) |
| `pit config [--section forecasting]` | effective configuration and version hashes |

Global options: `--db PATH`, `--config extra.yaml`, `--log-level DEBUG`.
`python -m pit …` works without installing the console script.

## 3. Dashboard

```bash
.venv/bin/streamlit run pit/dashboard/app.py
# a different database:
PIT_DATABASE_PATH=var/demo.db .venv/bin/streamlit run pit/dashboard/app.py
```

The dashboard reads from the same database as the scheduler, so run both side by side.
To open a backtest run directly, use `http://host:8501/?run=<run_id>`.

## 4. Unattended operation

`pit run` starts the scheduler:

* **ingest**: every 60 s;
* **forecast**: 5 s after each 15-minute slot during market hours, after a fresh sync;
* **evaluate**: every 5 minutes;
* **paper**: every minute;
* **heartbeat**: every minute;
* **missed-slot detection**: hourly;
* **job-log pruning**: daily.

It is safe to stop at any time (SIGINT or SIGTERM finish the current job). On restart:

* candles are caught up incrementally (feed status shows the outage);
* slots that passed while it was down are recorded as **missed**, never back-filled;
* duplicate forecasts are impossible (there is a unique slot key);
* a lock file (`<db>.scheduler.lock`) prevents two schedulers on one database;
* edits to `config/default.yaml` or `config/local.yaml` are picked up automatically; a
  forecasting change creates a new config version.

### systemd (Linux VPS)

`/etc/systemd/system/pit-scheduler.service`:

```ini
[Unit]
Description=PIT scheduler (research / paper trading only)
After=network-online.target

[Service]
User=pit
WorkingDirectory=/opt/pit
EnvironmentFile=/opt/pit/.env
ExecStart=/opt/pit/.venv/bin/python -m pit run
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/pit-dashboard.service`:

```ini
[Unit]
Description=PIT dashboard
After=network-online.target

[Service]
User=pit
WorkingDirectory=/opt/pit
ExecStart=/opt/pit/.venv/bin/streamlit run pit/dashboard/app.py --server.port 8501 --server.address 127.0.0.1
Restart=always

[Install]
WantedBy=multi-user.target
```

Expose the dashboard only behind an authenticating reverse proxy (or an SSH tunnel). It
has no login of its own and its Settings page can change the configuration.

### Cron alternative

```cron
# 10 s after every 15-minute slot, plus a minute-level sync for evaluation and paper trading
*/15 * * * *  cd /opt/pit && sleep 10 && .venv/bin/python -m pit run --once --no-lock
*    * * * *  cd /opt/pit && .venv/bin/python -m pit sync && .venv/bin/python -m pit evaluate && .venv/bin/python -m pit paper
```

## 5. Monitoring

The **System Health** page (or `pit status`) shows:

* the scheduler heartbeat;
* feed status and latency;
* gaps and data-quality events;
* job runs and failures;
* API errors;
* missed slots;
* database integrity and size;
* LLM usage and month-to-date cost against the US$150 budget.

What to check:

* **Heartbeat older than about 3 minutes**: the scheduler is not running.
* **Feed `stale` or `down` during market hours**: provider or network problem. New
  forecasts will be WAIT `data_stale` until data resumes.
* **Missed slots growing**: the host is overloaded or the process is restarting
  repeatedly. Check `var/logs/pit.log`.

Logs go to the console and to `var/logs/pit.log` (rotated, 5 × 5 MB).

## 6. Backups and data

* `pit backup` writes a consistent copy to `var/backups/`. Schedule it daily and copy it
  off the host.
* `var/` (databases, logs, backups) is git-ignored. Never commit it.
* Forecast and evaluation records are permanent by design. To start fresh, use a new
  database file rather than deleting rows.

## 7. Switching to real data (needs approval)

1. Pick a provider (see `docs/providers.md`) and get approval for any account or cost.
2. Put the credentials in `.env`, then set `data.provider` (on the Settings page or in
   `config/local.yaml`).
3. Run `pit ingest --days 100` to backfill. 100 days covers the 12h warm-up.
4. Restart `pit run`.

Candles from different sources are stored separately, and forecasts record their
provider and data mode.

## 8. Costs

Most computation runs locally. Recurring costs are only:

* an approved data subscription (enter it under Settings → Costs for reporting);
* optional LLM calls, capped by `llm.monthly_budget_usd`. Models without verified
  pricing are never called.

The target is under US$150 per month in total.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `provider 'x' needs the environment variable …` | credentials missing from `.env`, or switch back to `synthetic` |
| `another scheduler holds …lock` | a scheduler is already running on that database |
| All forecasts WAIT `insufficient_data` | not enough history; ingest at least `data.history_days` |
| All forecasts WAIT `data_stale` | the feed is behind; check `pit sync` output and System Health |
| Dashboard shows nothing | wrong database; check the sidebar path or set `PIT_DATABASE_PATH` |
