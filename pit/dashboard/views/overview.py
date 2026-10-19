"""Dashboard (home): price, feed status, latest forecast, regime, recent performance, health."""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import streamlit as st

from pit.analytics import metrics
from pit.app.health import system_health
from pit.common.timeutils import parse_ts, to_epoch_min, utcnow
from pit.dashboard import charts, data
from pit.dashboard.components import (
    data_mode_banner,
    direction_badge,
    fmt_ci,
    fmt_num,
    fmt_pct,
    fmt_price,
    heuristic_note,
    show_chart,
    status_badge,
)


def render() -> None:
    ctx = data.ctx()
    origin, run_id = data.scope()
    source = data.primary_source()
    instrument = ctx.settings.instrument.symbol
    st.title("Dashboard")
    fc = data.forecasts(data.key(), origin, run_id)
    modes = set(fc["data_mode"].unique()) if not fc.empty else set()
    if data.source_mode(source):
        modes.add(data.source_mode(source))
    data_mode_banner(modes, "Backtest replay" if origin == "backtest" else "")

    # ------------------------------------------------------------------ price + feed
    latest = ctx.candles.load(instrument, source, limit_last=1) if source else pd.DataFrame()
    feeds = {(f["instrument"], f["source"]): f for f in ctx.ops.feed_status()}
    feed = feeds.get((instrument, source), {})
    c1, c2, c3, c4, c5 = st.columns(5)
    if not latest.empty:
        row = latest.iloc[-1]
        age = (utcnow().timestamp() - (int(row["ts_min"]) + 1) * 60) / 60
        c1.metric("Gold price (mid)", fmt_price(row["close"]), help=f"Last completed 1-minute candle from {source}")
        c2.metric("Bid", fmt_price(row["bid_close"]), help=f"Spread source: {row['spread_source']}")
        c3.metric("Ask", fmt_price(row["ask_close"]), help=f"Spread source: {row['spread_source']}")
        c4.metric("Spread (USD/oz)", fmt_price(row["spread"], 2), help=f"Spread source: {row['spread_source']}")
        c5.metric("Data age", f"{age:,.0f} min", help=f"Feed status: {feed.get('status', 'unknown')}")
        st.caption(f"Feed: {status_badge(feed.get('status', 'unknown'))} · source `{source}` · quote convention "
                   f"`{row['quote_convention']}` · data mode **{row['data_mode']}**")
    else:
        st.info("No candles stored yet.")

    # ------------------------------------------------------------------ latest forecast
    st.subheader("Latest forecast")
    det = fc[fc["model_type"] == "deterministic"] if not fc.empty else fc
    if det.empty:
        st.info("No forecasts in this scope yet.")
    else:
        f = det.iloc[0]
        a, b, c, d, e = st.columns(5)
        a.metric("Direction", direction_badge(f["direction"]),
                 help=f"Candidate: {f['candidate_direction']}")
        b.metric("Composite score", fmt_num(f["composite_score"], 3), help="Range −1…+1; |S| ≥ 0.35 required")
        c.metric("Target", fmt_price(f["target_price"]),
                 help=f"Distance {fmt_num(f['target_distance'])} USD/oz from reference {fmt_price(f['reference_price'])}")
        d.metric("Heuristic P-HIT", fmt_pct(f["p_hit"]) if pd.notna(f["p_hit"]) else "–")
        e.metric("Confidence", f["confidence"])
        reasons = json.loads(f["wait_reasons_json"] or "[]")
        st.caption(
            f"Slot {f['slot_ts']} · horizon {f['horizon_minutes']} min · expires {f['expiry_ts']} · "
            f"regime **{f['regime']}** (vol {f['vol_regime']}, trend {f['trend_regime']}) · session {f['session']} · "
            f"outcome **{f['outcome_status'] or 'pending'}**"
            + (f" · WAIT reasons: {', '.join(reasons)}" if reasons else "")
        )
        heuristic_note()
        llm = fc[(fc["model_type"] == "llm_assisted") & (fc["parent_forecast_id"] == f["forecast_id"])]
        if not llm.empty:
            st.caption(f"LLM-assisted variant: {direction_badge(llm.iloc[0]['direction'])} "
                       f"({llm.iloc[0]['llm_model']}, prompt {llm.iloc[0]['llm_prompt_version']})")

        # Recent price with forecasts
        end_min = int(f["slot_min"]) + int(f["horizon_minutes"]) + 30
        candles = data.candles(data.key(), instrument, source, end_min - 6 * 60, end_min) if source else pd.DataFrame()
        if not candles.empty:
            fig, table = charts.price_chart(candles, "5m", det, show_wait=False, height=420,
                                            title=f"{instrument} — last 6 hours around the latest forecast (5m)")
            show_chart(fig, table, key="overview_price")

    # ------------------------------------------------------------------ recent performance
    st.subheader("Recent performance")
    if not det.empty:
        if origin == "live":
            cutoff = to_epoch_min(utcnow() - timedelta(days=7))
            recent = det[det["slot_min"] >= cutoff]
            label = "last 7 days"
        else:
            recent, label = det, "whole backtest run"
        s = metrics.summarise(recent, ctx.settings.forecasting.interval_minutes)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(f"Directional forecasts ({label})", f"{s.get('n_directional', 0)} / {s.get('n_forecasts', 0)}",
                  help="Directional / all forecasts (WAIT is a legitimate outcome)")
        p2.metric("Target-hit rate", fmt_pct(s.get("target_hit_rate")), help=fmt_ci(s.get("target_hit_rate_ci")))
        p3.metric("Target before mirror", fmt_pct(s.get("target_before_mirror_rate")),
                  help="Share of forecasts whose target was touched before the equidistant opposite target. "
                       "50% = no directional skill. " + fmt_ci(s.get("target_before_mirror_ci")))
        p4.metric("Expectancy (USD/oz/trade)", fmt_num(s.get("expectancy_per_oz"), 3),
                  help="Hypothetical bracket trade after spread and slippage")
        st.caption("Overlapping forecasts are not independent evidence; see Performance for bootstrap intervals.")

    # ------------------------------------------------------------------ health summary
    st.subheader("System health")
    h = system_health(ctx)
    h1, h2, h3, h4 = st.columns(4)
    sched = next((b for b in h["heartbeats"] if b["component"] == "scheduler"), None)
    h1.metric("Scheduler", "running" if sched and sched["age_seconds"] < 180 else "not running",
              help=f"Last heartbeat {sched['last_beat'] if sched else 'never'}")
    h2.metric("API errors (24h)", h["api_errors_24h"])
    h3.metric("Missed slots (7d)", h["missed_slots_7d"])
    h4.metric("Month cost (USD)", fmt_num(h["monthly_cost"]["total_usd"], 2),
              help=f"Budget {h['monthly_cost']['budget_usd']} USD")
    last = h["last_live_forecast"]
    st.caption(f"Last live forecast: {last['created_at'] if last else 'none'}")
