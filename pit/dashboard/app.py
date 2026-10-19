"""PIT Streamlit dashboard.

Run with:  streamlit run pit/dashboard/app.py
(optionally:  PIT_DATABASE_PATH=var/demo.db streamlit run pit/dashboard/app.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run pit/dashboard/app.py` without installing the package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from pit.dashboard import data  # noqa: E402
from pit.dashboard.views import (  # noqa: E402
    backtests,
    health,
    history,
    live_chart,
    overview,
    paper,
    performance,
    settings,
)

st.set_page_config(page_title="PIT — Predictive Intelligence Trading", page_icon="📈", layout="wide")


def sidebar() -> None:
    st.sidebar.title("PIT")
    st.sidebar.caption("Predictive Intelligence Trading · XAU/USD research platform · paper trading only")
    ctx = data.ctx()
    runs = data.runs(data.key())
    options = ["live"] + [r["run_id"] for r in runs]
    labels = {"live": "Live forecasts"}
    for r in runs:
        tag = "OOS" if r["validation_type"] == "out_of_sample" else "exploratory"
        labels[r["run_id"]] = f"Backtest: {r['name']} ({r['start_ts'][:10]}→{r['end_ts'][:10]}, {tag}, {r['data_mode']})"
    # A run can be preselected with ?run=<run_id> (shareable links).
    current = st.session_state.get("scope_choice", st.query_params.get("run", "live"))
    if current not in options:
        current = "live"
    choice = st.sidebar.selectbox("Forecast scope", options, index=options.index(current),
                                  format_func=lambda o: labels.get(o, o), key="scope_choice")
    st.session_state["origin"] = "live" if choice == "live" else "backtest"
    st.session_state["run_id"] = "" if choice == "live" else choice
    if st.sidebar.button("↻ Refresh data", width="stretch"):
        data.clear()
        st.rerun()
    st.sidebar.divider()
    st.sidebar.caption(f"Database: `{ctx.db.path}`")
    st.sidebar.caption(f"Provider: `{ctx.settings.data.provider}` · config `{ctx.settings.forecast_config_version}`")
    st.sidebar.caption("No brokerage connection exists. No real orders are ever placed.")


sidebar()

pages = [
    st.Page(overview.render, title="Dashboard", icon="🏠", default=True),
    st.Page(live_chart.render, title="Live Chart", icon="🕯️", url_path="chart"),
    st.Page(history.render, title="Prediction History", icon="🗂️", url_path="history"),
    st.Page(performance.render, title="Performance", icon="📊", url_path="performance"),
    st.Page(paper.render, title="Paper Trading", icon="💼", url_path="paper"),
    st.Page(backtests.render, title="Backtests", icon="🧪", url_path="backtests"),
    st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
    st.Page(health.render, title="System Health", icon="🩺", url_path="health"),
]
st.navigation(pages).run()
