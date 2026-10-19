"""Feeds, data quality, scheduler, API errors, database and cost."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pit.app.health import system_health
from pit.dashboard import data
from pit.dashboard.components import fmt_num, status_badge


def render() -> None:
    ctx = data.ctx()
    st.title("System Health")
    h = system_health(ctx)

    k = st.columns(5)
    sched = next((b for b in h["heartbeats"] if b["component"] == "scheduler"), None)
    k[0].metric("Scheduler", "running" if sched and sched["age_seconds"] < 180 else "not running",
                help=f"Last heartbeat: {sched['last_beat'] if sched else 'never'}")
    last = h["last_live_forecast"]
    k[1].metric("Last live forecast", last["slot_ts"][5:16] if last else "–")
    k[2].metric("API errors (24h)", h["api_errors_24h"])
    k[3].metric("Missed slots (7d)", h["missed_slots_7d"])
    k[4].metric("Data gaps (7d)", f"{h['data_gaps_7d']['gaps']} / {h['data_gaps_7d']['minutes']} min")

    st.subheader("Data feeds")
    feeds = pd.DataFrame(h["feeds"])
    if not feeds.empty:
        feeds.insert(0, "state", feeds["status"].map(status_badge))
        feeds["latency_min"] = (feeds["latency_seconds"] / 60).round(1)
        st.dataframe(feeds.drop(columns=["latency_seconds"]), hide_index=True, width="stretch")
        st.caption("Latency = age of the latest completed candle. While a feed is stale or down, new live forecasts "
                   "are WAIT (data_stale) and missing prices are never fabricated.")
    else:
        st.caption("No feed activity recorded yet.")
    srcs = pd.DataFrame(ctx.candles.sources())
    if not srcs.empty:
        st.dataframe(srcs, hide_index=True, width="stretch")

    st.subheader("Missing data")
    gaps = pd.DataFrame(ctx.candles.gaps(limit=200))
    st.dataframe(gaps, hide_index=True, width="stretch") if not gaps.empty else st.caption("No gaps recorded.")
    q = pd.DataFrame(ctx.ops.recent_quality_events(200))
    if not q.empty:
        with st.expander(f"Data-quality events ({len(q)} most recent)"):
            st.dataframe(q, hide_index=True, width="stretch")

    st.subheader("Scheduler")
    hb = pd.DataFrame(h["heartbeats"])
    if not hb.empty:
        st.dataframe(hb, hide_index=True, width="stretch")
    jobs = pd.DataFrame(ctx.ops.recent_jobs(200))
    if not jobs.empty:
        summary = jobs.groupby(["job", "status"]).size().unstack(fill_value=0).reset_index()
        st.dataframe(summary, hide_index=True)
        with st.expander("Recent job runs"):
            st.dataframe(jobs, hide_index=True, width="stretch")
    missed = pd.DataFrame(ctx.ops.missed_slots(100))
    if not missed.empty:
        with st.expander(f"Missed forecast slots ({len(missed)})"):
            st.caption("Slots that passed without a forecast (downtime, outage, lateness). They are recorded, never "
                       "back-filled.")
            st.dataframe(missed, hide_index=True, width="stretch")

    st.subheader("API errors")
    errs = pd.DataFrame(ctx.ops.recent_api_errors(100))
    st.dataframe(errs, hide_index=True, width="stretch") if not errs.empty else st.caption("None recorded.")

    st.subheader("Database")
    db = h["database"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Integrity check", db["integrity"])
    c2.metric("Size", f"{db['size_bytes'] / 1e6:,.1f} MB")
    c3.metric("Forecasts stored", db["counts"]["forecasts"])
    st.dataframe(pd.DataFrame([db["counts"]]), hide_index=True)
    if st.button("Verify forecast content hashes"):
        res = ctx.forecasts.verify_hashes()
        if res["mismatched"]:
            st.error(f"{len(res['mismatched'])} forecasts fail hash verification: {res['mismatched'][:20]}")
        else:
            st.success(f"All {res['checked']} forecast records match their content hashes.")

    st.subheader("Monthly operating cost")
    cost = h["monthly_cost"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM (USD)", fmt_num(cost["llm_usd"], 2))
    c2.metric("Data subscription (USD)", fmt_num(cost["data_subscription_usd"], 2))
    c3.metric("Total (USD)", fmt_num(cost["total_usd"], 2))
    c4.metric("Budget (USD)", fmt_num(cost["budget_usd"], 0))
    if cost["api_usage"]:
        st.dataframe(pd.DataFrame(cost["api_usage"]), hide_index=True)
