# Research protocol

**Research question:** can a transparent, repeatable quantitative (and optionally
AI-assisted) method identify short-term XAU/USD moves reliably enough to produce
**positive simulated expectancy after realistic transaction costs**?

PIT does not assume the answer is yes. This document describes how to reach an honest
answer.

## 1. Evidence hierarchy (weakest to strongest)

1. **Synthetic data.** Tests the software only. The synthetic generator embeds a mild
   trend process, so "edges" on synthetic data are artefacts and prove nothing.
2. **Exploratory backtests.** Useful for understanding behaviour. Once you have looked
   at a period, it is in-sample for every later configuration.
3. **Pre-registered out-of-sample backtests.** Run `pit declare` (config hash, holdout
   window, hypothesis), then one `pit backtest --validation out_of_sample --declaration
   …`. Each declaration can be used once. Re-running a holdout makes it in-sample.
4. **Live forward testing.** Forecasts are recorded before outcomes exist. This is the
   only evidence that rules out every form of look-ahead, including an LLM's knowledge
   of historical prices.

## 2. What counts as skill

* **Not** the raw target-hit rate. A 0.75 × ATR target over 60 minutes is touched about
  85–90% of the time by a random walk. The reported `p₀` (the Brownian touch
  probability) makes that explicit.
* **Directional skill** means the primary target is hit more often than the equidistant
  **mirror** target (`edge vs mirror`, with a paired block-bootstrap interval), and
  `target before mirror` is above 50%.
* **Economic skill** means **expectancy per trade after spread and slippage > 0**, with a
  confidence interval that excludes zero, on executable bid/ask outcomes. Mid-price
  touches don't count, and ambiguous trades count as losses.
* **Calibration**: the Brier score is below the climatology reference, and the
  reliability diagram sits near the diagonal. P-HIT is an uncalibrated heuristic, so
  expect it to be miscalibrated at first.

## 3. Statistical cautions

* Forecasts overlap (a 15-minute interval with a 60-minute horizon). Outcomes are
  dependent, so naive standard errors are too small. Use the block-bootstrap intervals
  and the non-overlapping subsample reported on the Performance page.
* Breakdowns by session, regime and so on split the sample. Small groups are noisy, and
  scanning many breakdowns *will* produce false positives. Treat them as
  hypothesis-generating.
* Changing parameters after seeing results invalidates the out-of-sample status of the
  data you looked at. PIT records a new `config_version` for every change, so results
  are never pooled across configurations by accident.

## 4. Recommended workflow

1. Run on synthetic data: `pit demo`. Confirm the pipeline and dashboard behave.
2. With approved real data, run exploratory backtests on an *early* period to learn how
   the method behaves (coverage, WAIT reasons, target-distance curve, costs).
3. If, and only if, you change parameters, freeze the new configuration and **declare**
   a later, untouched holdout with a written hypothesis. Run it once.
4. Start the scheduler for live forward testing with the frozen configuration. Review
   monthly. Do not change the configuration mid-test. If you must, a new version starts
   a new test.
5. Enable the LLM (optional) only in live forward testing. Judge it with the *matched
   slots* comparison and the *LLM filter value* table, net of its API cost.

## 5. Claims policy

PIT and its documentation must not describe the method as profitable or reliable
without forward-testing evidence that meets §2 over a meaningful sample. Synthetic
results are always labelled as such and are never presented as trading performance.
