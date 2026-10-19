"""Backtest runs and out-of-sample research declarations."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from pit.backtesting.replay import BacktestRunner, BacktestValidationError
from pit.common.timeutils import iso
from pit.dashboard import data
from pit.dashboard.components import fmt_num, fmt_pct


def render() -> None:
    ctx = data.ctx()
    st.title("Backtests")
    st.caption(
        "Historical replay uses only data available at each simulated timestamp. Parameters are never optimised "
        "automatically. Comparing configurations on the same data is **exploratory**; only a pre-registered, "
        "single-use declaration makes a run **out-of-sample**. Live forward testing is the strongest evidence."
    )
    runs = data.runs(data.key())
    if runs:
        rows = []
        for r in runs:
            det = (r.get("summary") or {}).get("deterministic", {})
            edge = (r.get("summary") or {}).get("mirror_edge", {})
            paper = (r.get("summary") or {}).get("paper", {})
            rows.append({
                "run_id": r["run_id"], "name": r["name"], "status": r["status"], "validation": r["validation_type"],
                "data_mode": r["data_mode"], "source": r["data_source"], "window": f"{r['start_ts'][:16]} → {r['end_ts'][:16]}",
                "config": r["config_version"], "forecasts": det.get("n_forecasts"), "directional": det.get("n_directional"),
                "hit_rate": det.get("target_hit_rate"), "edge_vs_mirror": edge.get("edge"),
                "expectancy_usd_oz": det.get("expectancy_per_oz"), "paper_final_balance": paper.get("final_balance"),
                "created": r["created_at"][:19],
            })
        df = pd.DataFrame(rows)
        if (df["data_mode"] == "synthetic").any():
            st.error("🧪 Runs marked **synthetic** use generated data. They test the software, not the market.")
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"hit_rate": st.column_config.NumberColumn(format="%.3f"),
                                    "edge_vs_mirror": st.column_config.NumberColumn(format="%.3f"),
                                    "expectancy_usd_oz": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Select a run in the sidebar ('Forecast scope') to explore it on every page.")
    else:
        st.info("No backtests yet.")

    st.subheader("Run a backtest")
    sources = [s for s in data.sources(data.key()) if s["instrument"] == ctx.settings.instrument.symbol]
    if not sources:
        st.warning("Load candles first (`python -m pit ingest` or `python -m pit demo`).")
        return
    with st.form("run_backtest"):
        c1, c2, c3 = st.columns(3)
        src = c1.selectbox("Data source", [s["source"] for s in sources],
                           format_func=lambda x: f"{x} ({next(s['data_mode'] for s in sources if s['source'] == x)})")
        info = next(s for s in sources if s["source"] == src)
        last = pd.Timestamp(info["last_ts"]).date()
        start = c2.date_input("Start (UTC)", value=last - timedelta(days=7))
        end = c3.date_input("End (UTC, exclusive)", value=last)
        c4, c5, c6 = st.columns(3)
        name = c4.text_input("Name", value=f"backtest {start}→{end}")
        validation = c5.selectbox("Validation type", ["exploratory", "out_of_sample"])
        decls = [d for d in ctx.research.declarations() if not d["consumed_by_run"]]
        decl = c6.selectbox("Declaration (OOS only)", [None] + [d["declaration_id"] for d in decls])
        paper = st.checkbox("Simulate paper account", value=True)
        submitted = st.form_submit_button("Run backtest")
    if submitted:
        st.caption(f"Configuration under test: `{ctx.settings.forecast_config_version}` (edit on the Settings page).")
        with st.spinner("Replaying history…"):
            try:
                result = BacktestRunner(ctx, source=src).run(
                    pd.Timestamp(start, tz="UTC").to_pydatetime(), pd.Timestamp(end, tz="UTC").to_pydatetime(),
                    name=name, validation_type=validation, declaration_id=decl, paper=paper,
                )
            except BacktestValidationError as exc:
                st.error(str(exc))
                return
        data.clear()
        det = result.summary.get("deterministic", {})
        st.success(f"Run {result.run_id}: {result.forecasts} forecasts in {result.seconds}s · hit rate "
                   f"{fmt_pct(det.get('target_hit_rate'))} · expectancy {fmt_num(det.get('expectancy_per_oz'), 3)} USD/oz")

    st.subheader("Out-of-sample declarations")
    st.caption("Declare a configuration and a holdout period *before* looking at that period. Each declaration can be "
               "used by exactly one run; afterwards the period is in-sample.")
    with st.form("declare"):
        c1, c2 = st.columns(2)
        hs = c1.date_input("Holdout start", value=date.today() - timedelta(days=30))
        he = c2.date_input("Holdout end", value=date.today())
        hyp = st.text_area("Hypothesis (stated in advance)",
                           value="Directional forecasts hit their target more often than the mirror target, and "
                                 "expectancy after costs is positive.")
        ok = st.form_submit_button("Declare current configuration")
    if ok:
        did = ctx.research.declare(ctx.settings.forecast_config_version, iso(pd.Timestamp(hs, tz="UTC")),
                                   iso(pd.Timestamp(he, tz="UTC")), hyp)
        st.success(f"Declared {did} for config {ctx.settings.forecast_config_version}")
    decls = ctx.research.declarations()
    if decls:
        st.dataframe(pd.DataFrame(decls), hide_index=True, width="stretch")
