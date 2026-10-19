"""Interactive candlestick chart with forecast markers, target levels and outcomes."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pit.common.timeutils import to_epoch_min, utcnow
from pit.dashboard import charts, data
from pit.dashboard.components import data_mode_banner, show_chart

WINDOWS = {"6 hours": 360, "1 day": 1440, "3 days": 3 * 1440, "1 week": 7 * 1440, "1 month": 30 * 1440}
TF_DEFAULT = {"6 hours": "1m", "1 day": "5m", "3 days": "15m", "1 week": "60m", "1 month": "4h"}


def render() -> None:
    ctx = data.ctx()
    origin, run_id = data.scope()
    source = data.primary_source()
    instrument = ctx.settings.instrument.symbol
    st.title("Live Chart")
    if not source:
        st.info("No market data stored yet.")
        return

    fc = data.forecasts(data.key(), origin, run_id)
    modes = {data.source_mode(source)} | (set(fc["data_mode"].unique()) if not fc.empty else set())
    data_mode_banner(modes)

    # One control row above the chart.
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1.4, 1, 1])
    window = c1.selectbox("Window", list(WINDOWS), index=1)
    tfs = ["1m", "5m", "15m", "30m", "60m", "4h", "12h"]
    tf = c2.selectbox("Timeframe", tfs, index=tfs.index(TF_DEFAULT[window]))
    last_min = ctx.candles.latest_ts_min(instrument, source)
    if origin == "backtest" and not fc.empty:
        default_end = pd.Timestamp(int(fc["expiry_min"].max()) * 60, unit="s", tz="UTC")
    else:
        default_end = pd.Timestamp((last_min + 1) * 60, unit="s", tz="UTC") if last_min else pd.Timestamp(utcnow())
    end_date = c3.date_input("Window ends on", value=default_end.date())
    show_wait = c4.toggle("Show WAITs", value=False)
    show_targets = c5.toggle("Target levels", value=True)

    end_min = min(to_epoch_min(pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)), (last_min or 0) + 1)
    if pd.Timestamp(end_date, tz="UTC").date() == default_end.date():
        end_min = int(default_end.timestamp() // 60)
    start_min = end_min - WINDOWS[window]
    candles = data.candles(data.key(), instrument, source, start_min, end_min)
    if candles.empty:
        st.warning("No candles in this window (market closed or data missing).")
        return
    max_bars = 3000
    n_bars = (end_min - start_min) / charts.tf_minutes(tf)
    if n_bars > max_bars:
        st.caption(f"Too many bars for {tf} over {window}; showing a coarser timeframe would be clearer.")
    sub = fc[(fc["start_min"] >= start_min) & (fc["start_min"] < end_min) & (fc["model_type"] == "deterministic")] if not fc.empty else fc
    fig, table = charts.price_chart(candles, tf, sub, show_wait=show_wait, show_targets=show_targets,
                                    title=f"{instrument} {tf} · source {source}")
    show_chart(fig, table, key="live_chart")
    st.caption(
        "Candles are mid prices. ▲/▼ mark forecasts at their evaluation start and reference price; dotted lines are "
        "targets until expiry; filled circles mark the first target touch. Mid-price touches are not executable outcomes."
    )
    if not sub.empty:
        directional = sub[sub["direction"] != "WAIT"]
        st.caption(f"{len(sub)} forecasts in window · {len(directional)} directional · "
                   f"{int((directional['outcome_status'] == 'hit').sum())} targets touched")
