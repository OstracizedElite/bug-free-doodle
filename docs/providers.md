# Market-data provider research and recommendation

*Researched 19 October 2026. Provider websites could not be fetched from the build
environment (its network policy blocks them), so the figures below come from web search
results and third-party listings. **Check every price and entitlement on the provider's
own pricing page before you buy anything.** PIT will not buy a subscription or open an
account without your approval.*

## What PIT needs

| Requirement | Why |
|---|---|
| XAU/USD 1-minute candles, historical | Feature warm-up (about 60 days for the 12h EMA50), backtests |
| Completed-candle polling, or streaming | Live forecasts every 15 minutes |
| **Bid/ask** (ideally bid/ask OHLC) | Simulating executable trades: buy at the ask, sell at the bid |
| Tick quotes (optional) | Working out whether the stop or the target was hit first inside one minute |
| AUD/USD | Converting results into the AUD paper account |
| Australian access and licensing | The user is in Australia; research and personal use |

## Comparison

| | **OANDA v20 API** | **Twelve Data** | **Massive (formerly Polygon.io)** |
|---|---|---|---|
| XAU/USD | `XAU_USD` CFD (broker quote) | `XAU/USD` (aggregated reference) | `C:XAUUSD` forex aggregates |
| 1m history | Yes, M1 candles, 5,000 per request | Yes, `1min` time series, 5,000 rows per call | Yes, minute aggregates, paged |
| Live data | REST pricing and HTTP streaming | REST; WebSocket from the Pro plan | REST; WebSocket on the real-time tier |
| **Bid/ask** | **Yes: separate bid, mid and ask OHLC per candle** | **No** (one price per bar) | Aggregates no; last-quote and quotes endpoints yes |
| Tick quotes | Streaming prices | WebSocket price events (bid/ask not guaranteed) | Historical quotes endpoint |
| Rate limits | Generous (about 100+ requests/s documented for v20) | Credits per minute by plan (free tier is very limited) | 5 requests/min on the free tier; unlimited on paid tiers |
| Cost (unverified) | **Free with a practice (demo) account** | Free tier; paid plans roughly US$79–229/month reported, figures conflict | Forex tiers roughly US$29/month (delayed) to US$199/month (real-time and WebSocket) reported |
| Australia | OANDA Australia is ASIC-regulated; the v20 API needs a v20 account. Docs exclude only OANDA Global Markets and TMS Brokers from v20. **Confirm before opening an account** | Available worldwide; commercial use only on higher tiers | Available worldwide; business licensing is priced separately |
| Licensing | Data for your own account use; no redistribution | Personal or commercial depending on plan | Personal or business plans |
| PIT adapter | `pit/data/providers/oanda.py` (read-only), `streaming.OandaPriceStream` | `pit/data/providers/twelvedata.py`, `streaming.TwelveDataPriceStream` | `pit/data/providers/massive.py` |

Notes:

* **Price convention matters.** OANDA quotes are one broker's CFD prices. Twelve Data and
  Massive are aggregated reference prices. None of them equals Plus500's executable
  price. PIT stores `quote_convention` with every candle so these are never mixed up.
* Twelve Data and Massive aggregates carry no spread. PIT fills it from
  `data.default_spread` and labels it `spread_source='assumed'`. Executable-trade results
  on that data depend on the assumption.
* Massive's lower tiers are **delayed**, so its adapter defaults to `data_mode: delayed`.

## Recommendation (needs your approval)

1. **Development (now):** the synthetic provider. No cost, deterministic, clearly labelled.
2. **First real data, forward testing: OANDA practice account (free).** It is the only
   option here that gives observed bid/ask OHLC per minute at no cost. That matters most
   for the research question, because results depend on executable prices after costs.
   It needs (a) opening an OANDA demo account and (b) creating an API token, and both
   need your approval. Confirm in OANDA's portal that a v20 account is offered to
   Australian residents and that `XAU_USD` is tradeable on it.
3. **Independent cross-check, optional:** the Twelve Data free tier, to compare OANDA's
   prices with an aggregated reference. The cost stays at US$0.
4. **Tick-level history, only if justified later:** Massive's real-time forex tier
   (about US$199/month reported) or an equivalent. Consider it only if intrabar
   ambiguity turns out to be a material share of outcomes (PIT reports this share).

Expected recurring cost under this recommendation: **US$0/month** for data, well inside
the US$150/month target. Optional LLM costs are capped separately by
`llm.monthly_budget_usd`.

## Switching provider

```yaml
# config/local.yaml
data:
  provider: oanda          # synthetic | csv | twelvedata | oanda | massive
providers:
  oanda:
    environment: practice
```

Then put `OANDA_API_TOKEN` (and `OANDA_ACCOUNT_ID` for live quotes and streaming) in `.env`.
Candles from each source are stored separately (`candles.source`). Forecasts record their
provider, so results from different sources are never pooled by accident.

## Sources

* Twelve Data pricing and credits: https://twelvedata.com/pricing,
  https://support.twelvedata.com/en/articles/5615854-credits,
  https://support.twelvedata.com/en/articles/5335783-trial
* Massive / Polygon rebrand and tiers (third-party listings):
  https://apicostcalc.com/polygon.html, https://qveris.ai/guides/polygon-pricing-optimized/,
  https://costbench.com/software/financial-data-terminals/polygon-io/
* OANDA v20 API: https://developer.oanda.com/rest-live-v20/introduction/ ;
  OANDA Australia legal: https://oanda.com/au-en/legal/
