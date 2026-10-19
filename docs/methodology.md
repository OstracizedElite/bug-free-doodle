# PIT Version 1.0 forecasting methodology

> **Status: a testable research hypothesis, not a proven market edge.** Every threshold
> and weight below is an initial assumption. PIT exists to measure, honestly and
> repeatably, whether this method has any value after realistic costs. Do not assume
> that it does.

Model version `pit-det-1.0.0`. All parameters live in `config/default.yaml`. Each one is
hashed into the `config_version` stored on every forecast, so any change produces a
separately evaluated configuration.

Notation: `T` is the forecast slot (UTC, aligned to the 15-minute grid). `P` is the
reference price. A bar covering `[start, end)` is **completed** when `end ≤ T`.

---

## 1. Data used at time T

* Only **completed** bars are used. The partially formed current bar is never used.
* 1-minute candles are resampled into 5m, 15m, 30m, 60m, 4h and 12h bars aligned to the
  UTC epoch (so 4h and 12h bars align to UTC midnight). A higher-timeframe bar's OHLC
  comes only from its own minutes. Bars are resampled once over all available data and
  then filtered to `end ≤ T`, which is look-ahead safe.
* **Fixed lookback windows** (`features.lookback_bars`; for example 200 bars of 60m)
  mean a feature depends only on the most recent N completed bars. A forecast is
  therefore identical whether it is computed live from a short history or during replay
  over years of data. This is verified by tests.
* A timeframe with fewer than `min_bars` (60) completed bars is *unavailable*. The
  1m, 5m, 15m and 60m timeframes are *required*. If any is unavailable the forecast is
  WAIT (`insufficient_data`).
* **Reference price** `P` is the mid close of the last completed 1-minute candle
  (`mid_close_1m`). Its bid, ask, spread and spread provenance (observed, derived or
  assumed) are recorded with the forecast.

## 2. Features

### 2.1 Volatility
* **True range** `TRᵢ = max(Hᵢ − Lᵢ, |Hᵢ − Cᵢ₋₁|, |Lᵢ − Cᵢ₋₁|)`.
* **ATR(14)** is the simple mean of the last 14 true ranges (needs 15 bars). There is
  never a partial warm-up value.
* **σ₁ₘ** is the sample standard deviation of the last 60 one-minute log returns between
  *consecutive* minutes. Returns spanning a gap or a market close are excluded, and at
  least 30 are required.
* **Volatility ratio** = ATR(14) / mean TR over the last 96 bars. It is ≥ 1.5 when
  expanding and ≤ 0.67 when contracting.
* Also stored per timeframe: the rolling standard deviation and the realised volatility
  `√Σr²` of the last 12 bar returns, plus the high–low range of the last 12 bars, in USD
  and in ATR units.

### 2.2 Momentum (per timeframe `tf`, window `W = momentum_lookback_bars × tf`)
* `P_hist` is the close of the last completed `tf` bar whose end is at or before `T − W`.
* `W_eff` is the number of **open-market minutes** between that bar's end and `P`'s
  timestamp.
* `ΔP = P − P_hist`, `R = (P / P_hist − 1) × 100`.
* **Volatility-normalised momentum** `z = ln(P / P_hist) / (σ₁ₘ · √W_eff)`.
* **Directional consistency** = (number of up closes − number of down closes) / 12, over
  the last 12 bar-to-bar changes, in [−1, 1]. The up-fraction is stored too.

### 2.3 Trend
* EMA 9, 21 and 50 and SMA 20 on each timeframe. An EMA is computed over the lookback
  window, seeded with the SMA of the window's first `period` values:
  `EMAₜ = α·xₜ + (1 − α)·EMAₜ₋₁` with `α = 2 / (period + 1)`.
* **Separation** `sep = (EMA9 − EMA21) / ATR`.
* **Slope** `slope = (EMA21ₜ − EMA21ₜ₋₃) / (3 · ATR)`, in ATR per bar.
* **Relative position** `(P − EMA21) / ATR`.

### 2.4 Market structure (15m bars, last 96)
* **Swing high**: bar *i* whose high is strictly above the 2 preceding highs and at or
  above the 2 following highs. Swing lows mirror this. A swing is *confirmed* only once
  its 2 following bars have completed, so there is no future confirmation.
* **Swing pattern** = ½·sign(SH₁ − SH₂) + ½·sign(SL₁ − SL₂), using the last two confirmed
  swing highs and swing lows.
* **Breakout levels**: the most recent confirmed swing high and swing low. If none exists
  (common in persistent trends), the prior range extreme is used instead, excluding the
  most recent 3 bars.
* **Breakout signal**:
  * +1 if `P` is above the high level and −1 if below the low level;
  * otherwise −0.5 for a *failed upside breakout* (a later bar traded above the high
    level but price is back inside) and +0.5 for a failed downside breakout;
  * otherwise 0.
* **Structure score** = clip(0.5·pattern + 0.5·breakout, −1, 1).
* Also stored: nearest confirmed resistance above and support below (in USD and ATR),
  range position `(P − low) / (high − low)`, and *consolidating* (range of the last 12
  bars < 6 ATR).

### 2.5 Market regime (15m bars; explicit precedence)

| Condition | Label |
|---|---|
| fewer than `min_bars` bars | `uncertain` |
| volatility ratio ≥ 1.5 | `high_volatility` |
| EMA9 > EMA21 > EMA50, slope ≥ 0.05 and P > EMA21 | `uptrend` |
| EMA9 < EMA21 < EMA50, slope ≤ −0.05 and P < EMA21 | `downtrend` |
| volatility ratio ≤ 0.67 | `low_volatility` |
| \|slope\| < 0.05 and consolidating | `range_bound` |
| otherwise | `uncertain` |

The volatility regime (`low`, `normal`, `high`) and trend regime (`up`, `down`, `range`,
`none`) are stored separately for analysis.

## 3. Composite directional score

Each component is mapped deterministically into [−1, +1]:

| Component | Weight | Formula |
|---|---|---|
| `m1` 1-minute momentum | 0.10 | `tanh(z₁ₘ / 2)` |
| `m5` 5-minute momentum | 0.20 | `tanh(z₅ₘ / 2)` |
| `m15` 15-minute momentum | 0.25 | `tanh(z₁₅ₘ / 2)` |
| `trend60` 60-minute trend | 0.20 | `tanh(0.5·sep₆₀ / 1.0 + 0.5·slope₆₀ / 0.2)` |
| `ma_alignment` | 0.10 | mean over 5m, 15m and 60m of `[sign(P − EMA9) + sign(EMA9 − EMA21) + sign(EMA21 − EMA50)] / 3` |
| `structure` | 0.10 | structure score (§2.4) |
| `vol_consistency` (volatility-adjusted directional consistency) | 0.05 | signed efficiency ratio on 5m: `(Cₜ − Cₜ₋₁₂) / Σ|ΔC|` |

`S = Σ wᵢ · cᵢ`. Because the weights sum to 1, S lies in [−1, +1].

* **Conflict** = the total weight of components with |cᵢ| ≥ 0.10 whose sign opposes S.
* **Candidate direction** = sign(S): UP, DOWN, or NONE if S = 0.
* Initial rule: S ≥ +0.35 gives an UP candidate, S ≤ −0.35 a DOWN candidate, otherwise
  WAIT.

## 4. WAIT conditions

Every applicable reason is recorded. The first one listed is the *primary* reason.

| Reason | Rule |
|---|---|
| `market_closed` | the market is closed at `T−1` or `T` (no just-completed candle) |
| `data_stale` | `T − end(last 1m candle) > stale_after_seconds` (3 minutes) |
| `data_quality` | a candle flagged `suspect` starts within the last 30 minutes |
| `insufficient_data` | a required timeframe, σ₁ₘ, ATR or any component is unavailable |
| `event_restriction` | a configured, verified economic event's window `[t − 30, t + 30]` overlaps `[start, expiry)` |
| `session_close` | a market close falls inside `[start, expiry)` |
| `excessive_spread` | spread / target distance > 0.35 |
| `volatility_invalid` | regime volatility ratio > 3.0, or ATR₁ₘ / spread < 1.5 (too quiet to cover costs) |
| `conflicting_signals` | conflict > 0.40 |
| `score_below_threshold` | −0.35 < S < +0.35 |

**WAIT is a legitimate forecast outcome.** For WAITs caused by anything *other* than a
data-integrity reason (the first four rows), the candidate direction, targets and
P-HIT are still stored and evaluated. That makes the value of each filter measurable.
Integrity WAITs carry no candidate.

## 5. Targets

* **Primary distance** `D = max(0.75 × ATR14(1m), 2 × spread)`, in USD per troy ounce.
* UP: `target = P + D`. DOWN: `target = P − D`.
* **Alternative distances**: 0.5, 1.0, 1.5 and 2.0 × ATR14(1m), with no spread floor.
  All are evaluated.
* **Mirror target**: distance D in the *opposite* direction. This is the built-in
  control. A random walk touches nearby targets often, so directional skill means
  hitting the target more often than the mirror, not more often than 50%.
* **Hypothetical stop** (used only for executable evaluation) = `P ∓ 1.0 × D`.

Touches measured on mid prices are **not** executable outcomes (§7).

## 6. Heuristic target-hit probability (P-HIT), version `phit-heuristic-v1`

> **Uncalibrated heuristic.** It is stored with its formula and version, its calibration
> is measured, and it is never auto-adjusted.

```
σ_P   = σ₁ₘ × P                                   (USD per √minute)
p₀    = 2 · (1 − Φ(D / (σ_P · √H)))               driftless Brownian first-passage probability
A     = direction_sign × ma_alignment             trend alignment ∈ [−1, 1]
logit = logit(clip(p₀, 0.01, 0.99))
      + 2.0  · (|S| − 0.35)
      + 0.25 · A
      − 1.0  · (spread / D)
P-HIT = clip(sigmoid(logit), 0.02, 0.98)
```

**Confidence**: `high` if |S| ≥ 0.60 and P-HIT ≥ 0.60; `medium` if |S| ≥ 0.45; otherwise
`low`. A WAIT has confidence `none`.

## 7. Timing and immutability

* Data cut-off (`as_of_ts`) = `T`.
* `created_at` is the wall-clock time the record was produced. For backtests it is the
  simulated clock; those records are labelled `origin = 'backtest'` with a run id.
* `start_ts` = max(T, created_at rounded up to the next minute). Price moves before a
  forecast existed are never credited to it.
* `expiry_ts` = start + horizon (default 60 minutes; 15, 30 and 120 are supported).
* A scheduled slot that cannot be forecast within 90 s is recorded as **missed** and
  never forecast after the fact.
* Records are immutable: database triggers reject UPDATE and DELETE, and each record
  carries a SHA-256 content hash (`pit verify`).

## 8. Evaluation (`eval-v1`)

Evaluation uses 1-minute candles in `[start, expiry)`.

| Measure | Definition |
|---|---|
| Target touch | first candle whose mid high (UP) or mid low (DOWN) reaches the target; `minutes_to_touch` = minutes from start to the end of that candle |
| Miss / indeterminate | not touched. It counts as a miss only if at least 90% of expected open-market minutes have candles, otherwise `indeterminate` |
| Target first | target touched before the mirror target (`yes`, `no`, `ambiguous` if in the same candle, or `neither`) |
| MFE / MAE | maximum favourable and adverse mid excursion from P over the window |
| MAE before hit | adverse excursion up to and including the touch candle. It is flagged ambiguous when the touch candle itself holds the extreme, because the order within a candle is unknown |
| Directional accuracy | the mid close at expiry is beyond P in the forecast direction |
| **Executable trade** | bracket order entered at the open of the first candle: long at **ask** + slippage, short at **bid** − slippage. Take-profit is at the target and the stop at the hypothetical stop, both checked on the **exit side** (long exits at the bid, short at the ask). A candle that opens through the stop fills at the open. If one candle reaches both levels the outcome is **`ambiguous`**: never a success, with P&L booked as the stop (conservative). Stored ticks resolve the order when available. With no exit before expiry, the trade closes at the last close |

## 9. Paper trading

Paper trading is kept separate from forecasts. Defaults:

* **Account**: A$50,000; leverage 20:1 (the ASIC retail limit for gold); margin
  close-out at 50%.
* **Sizing**: 0.5% of equity at risk to a stop at 1.0 × D. Notional is capped at 5 ×
  equity and limited by free margin.
* **Costs and limits**: slippage of 0.05 USD/oz per fill; overnight financing of 7% p.a.
  long and 3% p.a. short, charged per 17:00 New York rollover crossed.
* **Halts**: maximum daily loss 2%; maximum drawdown 10% (halts trading); at most 1
  concurrent position.
* **Currency**: AUD conversion uses only an observed AUD/USD rate. Otherwise the trade
  is skipped at entry, or its USD P&L is held as unconverted at exit.

## 10. What is deliberately *not* done in V1

* There is no machine learning, model fitting or automatic parameter optimisation.
* P-HIT is never re-weighted from results.
* Parameter changes are manual. They are versioned automatically, and a change is only
  validated by evaluation on later, unseen data (`docs/research_protocol.md`).
