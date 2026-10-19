# Dashboard

```bash
.venv/bin/streamlit run pit/dashboard/app.py          # http://localhost:8501
PIT_DATABASE_PATH=var/demo.db .venv/bin/streamlit run pit/dashboard/app.py
```

The sidebar **Forecast scope** switches every page between live forecasts and any
backtest run. A run can also be opened by link: `/?run=<run_id>`.

## Pages

| Page | Contents |
|---|---|
| **Dashboard** | gold mid, bid, ask, spread and data age; feed status; latest forecast (direction, score, target, P-HIT, confidence, regime, WAIT reasons, outcome); 6-hour chart; recent performance; health summary |
| **Live Chart** | candlesticks from 1m to 12h over windows from 6 hours to 1 month; ▲/▼ forecast markers at evaluation start; dotted target lines until expiry; target-touch markers; optional WAITs |
| **Prediction History** | filters (direction, confidence, outcome, model, text search), CSV export, per-forecast detail: component chart, targets with outcomes, evaluation, provenance and hash, feature snapshot, LLM response, price window with stop and candidate target |
| **Performance** | headline metrics including the **edge vs the mirror target**; overlap caveat with block-bootstrap and independent-subsample figures; reliability diagram, Brier score and log loss; target-distance curve; breakdowns (direction, hour, session, volatility regime, regime, target distance, confidence, model version, horizon); cumulative P&L; model comparison (all and matched slots); LLM filter value; WAIT-filter value |
| **Paper Trading** | balance and equity (AUD), drawdown, halts, unconverted-FX warnings, equity and drawdown charts, win rate, average win and loss, profit factor, ambiguous exits, open positions, trade history, append-only transaction log |
| **Backtests** | runs with validation type and data mode; run a backtest; create and list out-of-sample declarations |
| **Settings** | provider, interval, horizons, thresholds, target multiple, conflict limit, LLM enable/provider/model/budget/frequency, paper-trading limits, cost reporting; validated before writing `config/local.yaml`; shows the resulting config version |
| **System Health** | scheduler heartbeat, feeds and latency, sources, gaps, data-quality events, job runs, missed slots, API errors, database integrity and size, hash verification, LLM usage, monthly cost against budget |

## Data labelling

Every page shows a banner for the kind of data in scope:

* 🧪 **SYNTHETIC**: generated prices, not evidence of trading performance;
* 🗄️ **HISTORICAL** (replay);
* ⏱️ **DELAYED**;
* 📡 **LIVE** (paper trading only).

## Visual conventions

The palette was validated with the bundled colour-vision checker; light and dark
surfaces both pass.

* **Polarity**: UP is blue `#2a78d6` and DOWN is red `#e34948` (a diverging pair that
  stays distinguishable for colour-blind readers). WAIT is neutral grey. Direction is
  also always carried by ▲/▼ marker shapes and text labels.
* **Categorical identity** uses fixed slots in order: blue, orange, aqua and so on.
  Status colours are reserved for health states and always come with an icon and a
  label.
* Marks:
  * every chart has one y-axis; equity and drawdown are separate charts;
  * 2px lines and hairline grids;
  * bars no wider than about 24px, with rounded data ends;
  * area fills at about 10% opacity.
* Every chart has a **table view** expander, so no value is reachable only by hovering.
