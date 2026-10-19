"""Searchable forecast history with per-forecast detail."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from pit.dashboard import charts, data
from pit.dashboard.components import data_mode_banner, direction_badge, fmt_num, fmt_pct, fmt_price, heuristic_note, show_chart

COLUMNS = [
    "slot_ts", "model_type", "horizon_minutes", "direction", "candidate_direction", "composite_score",
    "reference_price", "target_price", "target_distance", "p_hit", "confidence", "regime", "session",
    "primary_wait_reason", "outcome_status", "minutes_to_touch", "target_first", "trade_outcome",
    "trade_pnl_per_oz", "data_mode", "forecast_id",
]


def render() -> None:
    ctx = data.ctx()
    origin, run_id = data.scope()
    st.title("Prediction History")
    fc = data.forecasts(data.key(), origin, run_id)
    if fc.empty:
        st.info("No forecasts in this scope yet.")
        return
    data_mode_banner(fc["data_mode"].unique())

    # Filters (one row above the table).
    c1, c2, c3, c4, c5 = st.columns(5)
    dirs = c1.multiselect("Direction", ["UP", "DOWN", "WAIT"], default=["UP", "DOWN", "WAIT"])
    confs = c2.multiselect("Confidence", ["high", "medium", "low", "none"], default=["high", "medium", "low", "none"])
    statuses = ["hit", "miss", "indeterminate", "not_applicable", "pending"]
    outcome = c3.multiselect("Outcome", statuses, default=statuses)
    models = sorted(fc["model_type"].unique())
    model_sel = c4.multiselect("Model", models, default=models)
    search = c5.text_input("Search (id, reason, regime)")

    d = fc.copy()
    d["outcome_status"] = d["outcome_status"].fillna("pending")
    d = d[d["direction"].isin(dirs) & d["confidence"].isin(confs) & d["outcome_status"].isin(outcome)
          & d["model_type"].isin(model_sel)]
    if search:
        s = search.lower()
        mask = (d["forecast_id"].str.lower().str.contains(s, regex=False)
                | d["wait_reasons_json"].str.lower().str.contains(s, regex=False)
                | d["regime"].fillna("").str.lower().str.contains(s, regex=False))
        d = d[mask]
    st.caption(f"{len(d)} of {len(fc)} forecasts")
    st.dataframe(d[COLUMNS], width="stretch", hide_index=True, height=360,
                 column_config={"p_hit": st.column_config.NumberColumn("P-HIT (heuristic)", format="%.2f"),
                                "composite_score": st.column_config.NumberColumn("score", format="%.3f")})
    st.download_button("Download CSV", d[COLUMNS].to_csv(index=False), file_name="pit_forecasts.csv", mime="text/csv")

    # ------------------------------------------------------------------ detail
    st.subheader("Forecast detail")
    if d.empty:
        return
    options = d["forecast_id"].tolist()[:500]
    fid = st.selectbox("Forecast", options,
                       format_func=lambda x: f"{d.loc[d.forecast_id == x, 'slot_ts'].iloc[0]} · "
                                             f"{d.loc[d.forecast_id == x, 'model_type'].iloc[0]} · "
                                             f"{d.loc[d.forecast_id == x, 'direction'].iloc[0]} · {x}")
    f = d[d["forecast_id"] == fid].iloc[0]
    comps = json.loads(f["components_json"] or "{}")
    a, b, c, e = st.columns(4)
    a.metric("Direction", direction_badge(f["direction"]), help=f"candidate {f['candidate_direction']}")
    b.metric("Score", fmt_num(f["composite_score"], 3))
    c.metric("Target", fmt_price(f["target_price"]), help=f"stop (hypothetical) {fmt_price(f['stop_price'])}")
    e.metric("P-HIT", fmt_pct(f["p_hit"]) if pd.notna(f["p_hit"]) else "–")
    heuristic_note()
    left, right = st.columns(2)
    with left:
        if comps.get("components"):
            fig = charts.components_chart(comps["components"], comps.get("weights", {}))
            fig.update_layout(title="Score components")
            show_chart(fig, pd.DataFrame([{"component": k, "value": v, "weight": comps.get("weights", {}).get(k)}
                                          for k, v in comps["components"].items()]), key=f"comp_{fid}")
        reasons = json.loads(f["wait_reasons_json"] or "[]")
        st.markdown("**WAIT reasons:** " + (", ".join(reasons) if reasons else "none"))
        st.markdown(
            f"**Record:** created {f['created_at']} · as-of {f['as_of_ts']} · evaluation window "
            f"{f['start_ts']} → {f['expiry_ts']} · model `{f['model_version']}` · config `{f['config_version']}` · "
            f"provider `{f['provider']}` ({f['quote_convention']}, {f['data_mode']}) · spread source "
            f"`{f['spread_source']}` · snapshot `{(f['snapshot_id'] or '')[:12]}` · hash `{f['content_hash'][:12]}`"
        )
        if f["model_type"] == "llm_assisted":
            st.markdown(f"**LLM:** {f['llm_provider']} / {f['llm_model']} · prompt {f['llm_prompt_version']} · "
                        f"parent `{f['parent_forecast_id']}`")
            call = ctx.db.query_one("SELECT response_json, status, cost_usd, latency_ms FROM llm_calls WHERE call_id = ?",
                                    (f["llm_call_id"],))
            if call:
                with st.expander("LLM response"):
                    st.json(json.loads(call["response_json"]) if call["response_json"] else {"status": call["status"]})
    with right:
        targets = pd.DataFrame([t.model_dump() for t in ctx.forecasts.targets(fid)])
        if not targets.empty:
            tev = ctx.db.frame("SELECT label, hit, first_touch_ts, minutes_to_touch FROM target_evaluations "
                               "WHERE forecast_id = ?", (fid,))
            if not tev.empty:
                targets = targets.merge(tev, on="label", how="left")
            st.markdown("**Targets**")
            st.dataframe(targets.drop(columns=["forecast_id"]), hide_index=True, width="stretch")
        st.markdown("**Evaluation**")
        ev = {k: f[k] for k in ("outcome_status", "evaluated_at", "first_touch_ts", "minutes_to_touch", "target_first",
                                "mfe", "mae", "mae_before_hit", "mae_before_hit_ambiguous", "close_at_expiry",
                                "direction_correct", "trade_outcome", "trade_pnl_per_oz", "trade_ambiguous", "coverage")}
        st.dataframe(pd.DataFrame([ev]).T.rename(columns={0: "value"}).astype(str), width="stretch")
        if f["snapshot_id"]:
            snap = ctx.forecasts.snapshot(f["snapshot_id"])
            if snap:
                with st.expander("Feature snapshot (inputs at forecast time)"):
                    st.json(snap["features"])

    # Price around the forecast
    source = f["provider"]
    candles = data.candles(data.key(), f["instrument"], source, int(f["slot_min"]) - 120, int(f["expiry_min"]) + 30)
    if not candles.empty:
        one = d[d["forecast_id"] == fid]
        fig, table = charts.price_chart(candles, "1m", one, show_wait=True, height=420,
                                        title="Price before and after the forecast (1m)")
        if pd.notna(f["target_price"]) and f["direction"] == "WAIT":
            fig.add_hline(y=f["target_price"], line_color="#8a8984", line_width=1, line_dash="dot",
                          annotation_text=f"candidate target ({f['candidate_direction']})",
                          annotation_position="top right")
        if pd.notna(f["stop_price"]):
            fig.add_hline(y=f["stop_price"], line_color="#8a8984", line_width=1,
                          annotation_text="hypothetical stop", annotation_position="bottom right")
        show_chart(fig, None, key=f"detail_price_{fid}")
