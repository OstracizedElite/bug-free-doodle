"""Forecast accuracy, calibration and simulated profitability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pit.analytics import metrics
from pit.dashboard import charts, data
from pit.dashboard.components import data_mode_banner, fmt_ci, fmt_num, fmt_pct, heuristic_note, show_chart

BREAKDOWNS = {
    "Direction": "direction",
    "Hour of day (UTC)": "hour_utc",
    "Trading session": "session",
    "Volatility regime": "vol_regime",
    "Market regime": "regime",
    "Target distance (USD/oz)": "target_bucket",
    "Confidence grade": "confidence",
    "Model version": "model_version",
    "Horizon (min)": "horizon_minutes",
}


def render() -> None:
    ctx = data.ctx()
    origin, run_id = data.scope()
    interval = ctx.settings.forecasting.interval_minutes
    st.title("Performance")
    fc = data.forecasts(data.key(), origin, run_id)
    if fc.empty:
        st.info("No forecasts in this scope yet.")
        return
    data_mode_banner(fc["data_mode"].unique())

    c1, c2, c3 = st.columns(3)
    models = sorted(fc["model_type"].unique())
    model = c1.selectbox("Model", models, index=models.index("deterministic") if "deterministic" in models else 0)
    horizons = sorted(fc["horizon_minutes"].unique())
    horizon = c2.selectbox("Horizon (minutes)", horizons, index=horizons.index(60) if 60 in horizons else 0)
    versions = sorted(fc["config_version"].unique())
    cfg_sel = c3.multiselect("Config versions", versions, default=versions,
                             help="Each parameter change creates a new config version; compare them separately.")
    d = fc[(fc["model_type"] == model) & (fc["horizon_minutes"] == horizon) & fc["config_version"].isin(cfg_sel)]
    if d.empty:
        st.info("No forecasts match the selection.")
        return
    s = metrics.summarise(d, interval)
    te = data.target_evaluations(data.key(), origin, run_id)
    te = te[te["forecast_id"].isin(d["forecast_id"])] if not te.empty else te
    edge = metrics.mirror_edge(te, interval)

    # ------------------------------------------------------------------ headline
    st.subheader("Headline")
    k = st.columns(5)
    k[0].metric("Forecasts", s.get("n_forecasts", 0), help=f"WAIT: {s.get('n_wait', 0)}")
    k[1].metric("Directional coverage", fmt_pct(s.get("directional_coverage")),
                help=f"{s.get('n_directional', 0)} directional forecasts")
    k[2].metric("Target-hit rate", fmt_pct(s.get("target_hit_rate")), help=fmt_ci(s.get("target_hit_rate_ci")))
    k[3].metric("Mirror-target hit rate", fmt_pct(edge.get("mirror_hit_rate")),
                help="Hit rate of the equidistant target in the opposite direction - the no-skill control")
    k[4].metric("Edge vs mirror", fmt_pct(edge.get("edge")), help=fmt_ci(edge.get("edge_ci")))
    k = st.columns(5)
    k[0].metric("Target before mirror", fmt_pct(s.get("target_before_mirror_rate")),
                help=fmt_ci(s.get("target_before_mirror_ci")))
    k[1].metric("Direction correct at expiry", fmt_pct(s.get("directional_accuracy_at_expiry")),
                help=fmt_ci(s.get("directional_accuracy_ci")))
    k[2].metric("Executable success", fmt_pct(s.get("executable_success_rate")),
                help="Bracket order reached take-profit on bid/ask before the stop (ambiguous = not a success)")
    k[3].metric("Expectancy (USD/oz)", fmt_num(s.get("expectancy_per_oz"), 3),
                help="After spread and slippage. " + (f"95% CI {s['expectancy_ci'][0]:.3f} to {s['expectancy_ci'][1]:.3f}"
                                                      if s.get("expectancy_ci") and s["expectancy_ci"][0] is not None else ""))
    k[4].metric("Profit factor", fmt_num(s.get("profit_factor"), 2))
    st.caption(
        f"Mean minutes to touch {fmt_num(s.get('mean_minutes_to_touch'), 1)} · mean target {fmt_num(s.get('mean_target_distance'))} "
        f"USD/oz · MFE {fmt_num(s.get('mean_mfe'))} · MAE {fmt_num(s.get('mean_mae'))} · MAE before hit "
        f"{fmt_num(s.get('mean_mae_before_hit'))} · ambiguous trades {fmt_pct(s.get('ambiguous_trade_share'))} · "
        f"net {fmt_num(s.get('net_pnl_per_oz'))} USD/oz · max drawdown {fmt_num(s.get('max_drawdown_per_oz'))} USD/oz"
    )
    ind = s.get("independent", {})
    st.info(
        f"**Overlap caveat.** Forecasts every {interval} min with a {horizon}-min horizon overlap, so outcomes are "
        f"dependent. Intervals use a moving-block bootstrap (block = {s.get('bootstrap_block')}). On a non-overlapping "
        f"subsample (n = {ind.get('n', 0)}) the hit rate is {fmt_pct(ind.get('target_hit_rate'))} "
        f"(Wilson {fmt_ci(ind.get('target_hit_rate_wilson'))[7:]}) and expectancy {fmt_num(ind.get('expectancy_per_oz'), 3)} USD/oz."
    )

    # ------------------------------------------------------------------ calibration
    st.subheader("Probability calibration (heuristic P-HIT)")
    heuristic_note()
    dd = metrics.prepare(d)
    dd = dd[dd["is_directional"]]
    p = pd.to_numeric(dd["p_hit"], errors="coerce").to_numpy(dtype=float)
    y = dd["hit"].to_numpy(dtype=float)
    table = metrics.calibration_table(p, y, bins=10)
    left, right = st.columns([3, 2])
    with left:
        show_chart(charts.reliability_chart(table), table, key="reliability")
    with right:
        st.metric("Brier score", fmt_num(s.get("brier_score"), 4),
                  help=f"Lower is better. Climatology reference (always predicting the base rate): "
                       f"{fmt_num(s.get('brier_reference_climatology'), 4)}")
        st.metric("Log loss", fmt_num(s.get("log_loss"), 4))
        st.metric("Mean P-HIT vs observed", f"{fmt_pct(s.get('mean_p_hit'))} vs {fmt_pct(s.get('target_hit_rate'))}")
        fig = charts.bin_counts_chart(table)
        fig.update_layout(title="Forecasts per P-HIT bin")
        show_chart(fig, None, key="bins")

    # ------------------------------------------------------------------ target distances
    st.subheader("Target distance")
    tdt = metrics.target_distance_table(te)
    if not tdt.empty:
        show_chart(charts.target_distance_chart(tdt), tdt, key="target_dist")
        st.caption("Every forecast is evaluated at 0.5/1.0/1.5/2.0 × ATR as well as the primary target. Compare "
                   "with the mirror line: small targets are touched often even without directional skill.")

    # ------------------------------------------------------------------ breakdowns
    st.subheader("Breakdowns")
    dim_label = st.selectbox("Break down by", list(BREAKDOWNS))
    b = metrics.breakdown(d, BREAKDOWNS[dim_label])
    if not b.empty:
        l2, r2 = st.columns(2)
        with l2:
            show_chart(charts.bar_chart(b, BREAKDOWNS[dim_label], "hit_rate", f"Target-hit rate by {dim_label.lower()}"),
                       None, key="bd_hit")
        with r2:
            show_chart(charts.bar_chart(b, BREAKDOWNS[dim_label], "expectancy_per_oz",
                                        f"Expectancy (USD/oz) by {dim_label.lower()}", pct=False,
                                        reference=0.0, reference_label="break-even"), None, key="bd_exp")
        st.dataframe(b, hide_index=True, width="stretch")
        st.caption("Small groups are noisy; check n before drawing conclusions.")

    # ------------------------------------------------------------------ over time
    st.subheader("Performance over time")
    freq = st.radio("Period", ["D", "W"], horizontal=True, format_func=lambda x: {"D": "Daily", "W": "Weekly"}[x])
    ot = metrics.performance_over_time(d, freq)
    if not ot.empty:
        show_chart(charts.line_chart(ot, "period", "cum_pnl_per_oz", "Cumulative hypothetical P&L (USD per oz, 1 oz per trade)",
                                     "USD/oz"), ot, key="cum")

    # ------------------------------------------------------------------ model comparison / WAIT filter
    st.subheader("Model comparison")
    cmp = metrics.compare_models(fc[(fc["horizon_minutes"] == horizon) & fc["config_version"].isin(cfg_sel)], interval)
    st.dataframe(cmp, hide_index=True, width="stretch")
    st.caption("Deterministic and LLM-assisted forecasts are stored separately. LLM-assisted results in backtests are "
               "not credible evidence (models may have seen historical prices); use live forward testing.")
    st.subheader("Value of WAIT filters")
    wf = metrics.wait_filter_analysis(d)
    if not wf.empty:
        st.dataframe(wf, hide_index=True, width="stretch")
        st.caption("WAIT forecasts keep a candidate direction where data integrity allows. If WAIT candidates perform "
                   "as well as directional forecasts, that filter is not adding value.")
